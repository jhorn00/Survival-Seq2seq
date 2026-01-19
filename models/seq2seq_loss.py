import torch
import torch.nn as nn

"""
Two types of loss.
1. Log-likelihood loss split into two parts:
    a. For uncensored data (patients who died), maximize prob that model predicts exact time bin Tau_t where patient died.
    b. For censored data (patients discharged alive), maximize prob that model predicts survival beyond last time bin observed Tau_t.
2. Ranking loss: for any pair of patients where one died before the other, the patient that died should have higher predicted risk at a time Tau_i.

Intuition:
p_k^j(τ_t) = predicted probability mass that patient j has event type k in time bin t (this is a discrete PDF, having event k in bin t)
cdfk_tau_t_given_xj = cumulative probability (that event k has occurred by time t)
u_uc = uncensored patients (patients which had mortality event)
u_c = censored patients (patients that were discharged alive)
u_a = acceptable pairs of patients for ranking (earlier vs later death)

Likelihood loss:
- For uncensored patients:
    - They died at discrete bin tj
    - Want to model high probability of dying in that bin so we minimize -log(p(tau_tj))
- For censored patients:
    - Alive at last observed time tj
    - We want them to survive beyond tj
    - Survival probability at tj is 1 - CDF(tj)
        - CDF(tj) is the probability that death has happened by tj
    - Minimize -log(1-CDF(tj))

Ranking loss (2019 methodology):
- u_a = pairs (i, j) where patient i dies before patient j (or j is censored)
- Intuition:
    - At time ti when patient i died:
        - i should look riskier than j
        - Riskier in survival means high event probability, meaning higher CDF
        - So we want CDF of tau_ti given x_i to be higher than it is given x_j

Likelihood loss means getting the absolute probabilities for event occurrence
Ranking loss means getting the rankings between patients right
"""

class SurvivalSeq2SeqLoss(nn.Module):
    def __init__(self, alpha=1.0, epsilon=1e-5):
        """
        Survival Seq2Seq paper (Eq. 3 & 4).
        L = L_l + alpha * L_r
        
        alpha (float): Weight for the ranking loss component.
        epsilon (float): Small number to prevent log(0).
        """
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, y_pred_pdf, y_true):
        """
        y_pred_pdf (torch.Tensor):
            Model predictions as probability density functions (PDFs). In other words, model's predicted probability that patient b has the event in time bin t (for each bin).
            Elements should sum to approximately 1 for a given patient.
            Shape: (batch, num_events, time_bins), or (B, 1, T) where:
                B = batch size (number of patients in this batch)
                1 = number of event types
                T = number of discrete time bins in horizon (198 for 4 hour increments over 33 days)
        y_true (torch.Tensor): The labels.
            Shape: (batch, 2), cols: [event_flag, event_time_bin]
                event_flag = mortality event, where 1 is uncensored (patient died in prediction window) and 0 is censored (patient alive at end of prediction window)
                event_time_bin = time bin index corresponding to the death time OR the censoring time if they were alive at the end of the prediction window
        """
        
        # We are only ever predicting a single event type, so we should eliminate the unnecessary dimension
        # (B, 1, T) -> (B, T)
        y_pred_pdf = y_pred_pdf.squeeze(1) 
        
        # Split up event flags and event times for patients
        # (B, 2) -> (B) and (B)
        event_flag = y_true[:, 0].float() # Died in the window or not - This has to be a float because of later operations
        event_time_bin = y_true[:, 1].long() # last bin we know their status

        ## Create masks for uncensored (death) and censored (no death)
        uncensored_mask = (event_flag == 1)
        censored_mask = (event_flag == 0)
        
        # Log-Likelihood Loss (L_l)

        ## First L_l term (uncensored, or mortality term)
        """
        Example:
        
        In this example, y_pred_pdf is still (B, T).
        Each row B is a batch, or patient.
        Each column T is a time bin where the probability of an event across a patient row should be roughly 1.
        y_pred_pdf =
        [
            [0.1, 0.2, 0.7],
            [0.3, 0.4, 0.3],
            [0.25, 0.25, 0.5]
        ]

        In this example, event_time_index is still (B, 1) after unsqueezing it.
        This specifies the time step where the event occurs (for uncensored entries) or the last known alive time (for censored entries).
        event_time_index =
        [
            [2],
            [1],
            [0]
        ]

        Gather.
        pdf_gathered =
        [
            [0.7],
            [0.4],
            [0.25]
        ]

        Squeeze it back down.
        pdf_at_event = [0.7, 0.4, 0.25]
        """

        ## What we want is p_k^j(τ_(t_j)) for each patient j, or the probability that patient j dies in the true event bin t_j
        ## pdf_at_event = p_k(τ_t | x), which means that pdf_at_event[j] is the model's predicted probability that the patient dies in the true event bin t_j given covariates x

        ### Expand from (B) to (B, 1) for torch.gather
        ### event_time_index = (B, 1) where each row j contains [t_j]
        event_time_index = event_time_bin.unsqueeze(1)
        

        ### Gather p_k^j(τ_(t_j)) for each patient j
        ### torch.gather selects pdf_gathered[j, 0] = y_pred_pdf[j, event_time_index[j, 0]]
        ### This is a column vector of the probabilities for each true bin for each patient j and is of shape (B, 1)
        pdf_gathered = torch.gather(
            y_pred_pdf, # (B, T)
            dim=1, # Gather from time bin dimension
            index=event_time_index # (B, 1)
        )
        ### Remove the extra dimension (B, 1) -> (B)
        pdf_at_event = pdf_gathered.squeeze(1)
        ## This is p_k^j(τ_(t_j)) for the L_l uncensored term
        ## pdf_at_event[j] is the model's belief in the event occurring exactly in the bin where patient j actually died (or last known bin for censored, but that will not be used)

        ### Uncensored Loss: -log(p_k(τ_t | x))
        # epsilon avoids log(0) issues
        loss_l_uncensored = -torch.log(pdf_at_event[uncensored_mask] + self.epsilon)


        ## Second L_l term (censored term)
        """
        For a censored patient j, all we know is that they were alive up until t_j

        The given log likelihood term for this is:
            L_l^(censored) = -log(1 - CDF(τ_(t_j) | x_j))

        Where:
            CDF(τ_t) = P(event has occurred by time t).
            S(t) = 1 - CDF(τ_t), or the survival probability. This is P(event has not yet occurred by time t).
        
            
        Example:
        y_pred_pdf is the same as before, with shape (B, T).
        y_pred_pdf =
        [
            [0.1, 0.2, 0.7],
            [0.3, 0.4, 0.3],
            [0.25, 0.25, 0.5]
        ]

        CDF is the cumulative sum across time bins.
        y_pred_cdf =
        [
            [0.1, 0.3, 1.0],
            [0.05, 0.10, 1.0],
            [0.6, 0.9, 1.0]
        ]

        event_time_bin is the last bin we know a patient's status.
        event_time_bin = [1, 2, 0]

        event_flag denotes whether a patient died in the window or not.
        event_flag = [0, 0, 1], so patients 0 and 1 are censored and patient 2 died (uncensored)

        We only care about censored patients (0 and 1) for this example because we want to calculate the censored log loss term.

        Step 1:
            Gather CDF at censor times t_j:

        event_time_index = [[1], [2], [0]]

        gather(y_pred_cdf, event_time_index) =
        [
            [0.3],
            [1.0],
            [0.6]    This one will not be used because patient is uncensored (died)
        ]

        Selecting only censored patient true time bin cdf
        cdf_at_censor = [0.3, 1.0]

        
        Step 2:
            Convert to survival probability:

        survival_prob = 1 - cdf_at_censor
            = [1 - 0.3, 1 - 1.0]
            = [0.7, 0.0]

        So,
            for patient 0, 1 - 0.3 = 0.7 (70% chance of being alive at censor time)
            for patient 1, 1 - 1.0 = 0.0 (0% chance of being alive at censor time)

        
        Step 3:
            Apply censored loss:

        loss_l_censored =
            [-log(0.7), -log(0.0 + epsilon)]
            = [0.357, 16.118]

        epsilon prevents things from breaking if we get 0s
        """

        ### CDF over time for each patient (probability event has happened by time t)
        y_pred_cdf = torch.cumsum(y_pred_pdf, dim=1)
        ### Calculate CDF: CDF(t) = sum_{i=0 to t} p(i)
        
        ### CDF value at specific censoring time t_j for each patient j
        ### cdf_at_censor = CDF(tau_t | x), which means that cdf_at_censor[j] is the model's predicted probability that the patient dies by the true event bin t_j given covariates x
        cdf_at_censor = torch.gather(
            y_pred_cdf,
            dim=1,
            index=event_time_bin.unsqueeze(1)
        ).squeeze(1)

        ### Censored Loss: -log(1 - CDF(tau_t | x))
        ### compute survival probability S(t_j) = 1 - CDF(t_j)
        ### (only for censored patients)
        survival_prob = 1.0 - cdf_at_censor[censored_mask]

        ### censored loss term: -log(S(t_j))
        ### epsilon avoids log(0) issues
        loss_l_censored = -torch.log(survival_prob + self.epsilon)


        ## combine uncensored and censored terms and average over batch
        ## Now we have L_l
        loss_l = (torch.sum(loss_l_uncensored) + torch.sum(loss_l_censored)) / event_flag.shape[0]

        
        ## Next we will need to calculate ranking loss (L_r)
        """
        This is supposed to be one of the key improvements from this paper. Using ranking loss from  
        Equation 4 (Jing et al. 2019) vs Equation 5 (Tjandra et al. 2021)

        Before, for every acceptable pair of patients (i, j) where i died before j or j was censored  
        we compare their CDFs at the event time of i. Each of these pairs will contribute one number to the loss.
        
        Effectively, Equation 4 would compute "When patient i died, does the model think patient i was more likely to die than patient j".
        Crucially, this ONLY checks the CDF for the time bin in which i died. Models could theoretically improve their loss by only  
        correcting one time step, which could result in more noisy curves and reduced overall accuracy. It is possible that just the  
        target bin is ranked properly but all other bins are mis-ranked.

        Equation 5 should reduce these risks and generate an overall more accurate ranking by checking every time bin.  
        This will require the ranking to be correct across the whole curve rather than just at a specific time bin.
        The expected effects are the following:
            - Models are not able to cheat by adjusting risk for a single point (this reminds me of overfitting or memorization)
            - The survival curves must now be globally coherent, not just good for a single section
                - It would prevent cases that are technically correct CDFs which are monotonically increasing where  
                  the CDF i was below j for all bins until it shoots up just before the measured bin
            - Resulting CDFs should be smoother, as well as being more discriminative than before

        Example (Equation 5):

        y_pred_cdf =
        [
            [0.05, 0.10, 0.40, 0.90],    uncensored (early death)
            [0.02, 0.05, 0.10, 0.20],    uncensored (later death)
            [0.01, 0.02, 0.04, 0.10]     censored
        ]

        Let's assume patient 0 died at bin 1, patient 1 died at bin 3, and patient 2 was censored at bin 3
        event_time_bin = [1, 3, 3]
        event_flag = [1, 1, 0]

        Acceptable pairs (U_a) are all (i, j) where
            - i is uncensored (event_flag[i] == 1)
            - event time for i is less than event time for j (event_time_bin[i] < event_time_bin[j])
        In this example, U_a = [(0, 1), (0, 2)]
            - patient 0 died before patient 1 and died before the censor time for patient 2
            - patient 1 died but at the same time as the censor time for patient 2

        Based on Equation 5, for all time bins t of all pairs (i, j), we compute
            diff[i, j, t] = CDF_i(t) - CDF_j(t)
            phi = exp(diff)
        then sum over each (i, j) in U_a and all time bins t.

        For (0, 1):
            exp(0.05 - 0.02) = exp(0.03) for t=0
            exp(0.10 - 0.05) = exp(0.05) for t=1
            exp(0.40 - 0.10) = exp(0.30) for t=2
            exp(0.90 - 0.20) = exp(0.70) for t=3

        For (0, 2):
            exp(0.05 - 0.01) = exp(0.04) for t=0
            exp(0.10 - 0.02) = exp(0.08) for t=1
            exp(0.40 - 0.04) = exp(0.36) for t=2
            exp(0.90 - 0.10) = exp(0.80) for t=3

        From here we sum phi(i, j, t) across all time bins t.
        That operation is followed by a mean (sum of all pairs (i, j) in U_a over the size of U_a).
        The resulting value should be multiplied by -1.
        """

        ## We will need a mask for the acceptable patient pairs
        ## patient i must be uncensored (deceased) and t_i < t_j
        ### event_time_bin is shape (B), which is a batch of the number of patients and represents the event time bin for each patient
        ### event_flag is also shape (B), and denotes the occurrence of death for the patient
        """
        Assuming B is 3 for a moment, (B, 1) would be something like:
        [
            [t for i=0],
            [t for i=1],
            [t for i=2]
        ]
        and (1, B) would be something like:
        [t for j=0, t for j=1, t for j=2]
        """
        event_time_i = event_time_bin.unsqueeze(1) # This becomes shape (B, 1)
        event_time_j = event_time_bin.unsqueeze(0) # This becomes shape (1, B)
        flag_i = event_flag.unsqueeze(1) # this becomes shape (B, 1)

        mask_i_uncensored = (flag_i == 1) # (B, 1)

        ### Gather acceptable pairs by time bin order (i before j)
        """
                    j=0     j=1     j=2
            i=0    t0<t0   t0<t1   t0<t2
            i=1    t1<t0   t1<t1   t1<t2
            i=2    t2<t0   t2<t1   t2<t2
        """
        mask_time_order = (event_time_i < event_time_j) # this becomes (B, B)

        ### Make acceptable pair mask and get the count of acceptable pairs
        acceptable_pair_mask = mask_i_uncensored & mask_time_order # result is still (B, B) since it applies to all columns
        num_pairs = acceptable_pair_mask.sum()

        ## Early termination if U_a is empty
        ### recall that y_pred_cdf is shape (B, T)
        if num_pairs == 0:
            loss_r = y_pred_cdf.new_tensor(0.0)
            total_loss = loss_l + self.alpha * loss_r
            return total_loss

        ### The original cdf matrix is (B, T), which has patients in rows and time in column. To compare i and j we need (B, B, T)
        ### i CDF representation just gets its rows wrapped in an extra list dimension
        #### each row is still a sequence of cdf[t] values for each time bin t of patient i
        ### j CDF representation gets the entire (B, T) matrix wrapped in an extra leading list dimension
        #### each row is still a sequence of cdf[t] values for each patient j — we only add a new axis so it can broadcast across i
        y_pred_cdf_i = y_pred_cdf.unsqueeze(1) # (B, T) -> (B, 1, T)
        y_pred_cdf_j = y_pred_cdf.unsqueeze(0) # (B, T) -> (1, B, T)
        diff = y_pred_cdf_i - y_pred_cdf_j # diff is (B, B, T)



        # # Paper ranking loss methodology

        # ### Convert CDF differences into positive ranking scores
        # ### same shape of (B, B, T)
        # phi = torch.exp(diff)

        # # ### Expand the mask so it can select across time dimension
        # # mask_3d = acceptable_pair_mask.unsqueeze(-1) # (B, B) -> (B, B, 1)
        # # ### Select only (i, j) pairs that are valid across all time bins t
        # # phi_pairs_time = phi[mask_3d] # (B, B, T) -> (num_pairs * T), but is maximally (B^2 * T) if all pairs were somehow acceptable
        # # ### Reshape to (num_pairs, T)
        # # ### each row is one acceptable pair (i, j) over all time bins
        # # phi_pairs_time = phi_pairs_time.view(num_pairs, -1)

        # phi_pairs_time = phi[acceptable_pair_mask]

        # ### Sum across time and take the mean relative to the pairs U_a, per Equation 5
        # loss_r = -phi_pairs_time.sum(dim=1).mean()




        # Experimental ranking loss methodology: limit exponential phi
        phi = torch.log(1 + torch.exp(-diff))
        phi_pairs_time = phi[acceptable_pair_mask]
        loss_r = phi_pairs_time.mean()




        # Combine Losses
        total_loss = loss_l + self.alpha * loss_r
        print(f"Total Loss: {total_loss}, Likelihood Loss: {loss_l}, Rank Loss: {loss_r}")
        return total_loss
