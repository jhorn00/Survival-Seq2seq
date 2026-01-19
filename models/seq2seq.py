import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "GRU-D"))

from GRUD import GRUD

'''
Assumptions from the paper:

Features (Covariates): 108

Encoder Length (Input Timesteps): Maximum of 60 timesteps.

Prediction Horizon: 33 days

Decoder Resolution (Time Bins): 4 hours

Competing Risks (K): 1 (in-hospital mortality)
'''

'''
Input Shape (for GRU-D layer):

Measurements: (batch_size, 60, 108)

Mask: (batch_size, 60, 108)

Delta (Time Gaps): (batch_size, 60, 108)
'''

class Encoder(nn.Module):
    """
    Setup with GRU-D layer and allows optional stacked RNN layers.
    Expects GRUD.forward input format: (batch, type_size, step_size, input_size)
      where type_size==4 and indices are:
        0 -> X (measurements) shape (batch, seq_len, input_size)
        1 -> X_last_obsv
        2 -> Mask
        3 -> Delta
    Returns:
      encoder_outputs: (batch, seq_len, enc_hidden_size)  -- outputs at each timestep
      final_hidden: (batch, enc_hidden_size) -- last hidden state (after stacked RNNs)
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, rnn_type, X_mean, output_last=False):
        super(Encoder, self).__init__()
        self.grud = GRUD(input_size=input_size,
                         cell_size=input_size,
                         hidden_size=input_size,
                         X_mean=X_mean,
                         output_last=output_last)

        # stacked RNNs (Not used in paper baseline)
        if num_layers > 1:
            if rnn_type == 'LSTM':
                self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers-1, batch_first=True, dropout=dropout)
            elif rnn_type == 'GRU':
                self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers-1, batch_first=True, dropout=dropout)
            else:
                raise ValueError('RNN type must be LSTM or GRU')
            self.output_dim = hidden_size
        else:
            self.rnn = None
            self.output_dim = input_size # fix grud problem

    def forward(self, grud_input):
        # grud_input shape: (batch, type_size=4, seq_len, feat)
        x = self.grud(grud_input)  # (batch, seq_len, grud_hidden)
        # pass through rnn if one is being used
        if self.rnn:
            x, _ = self.rnn(x)
        # out: (batch, seq_len, encoder_dim), last time step
        final_hidden = x[:, -1, :]  # (batch, encoder_dim)
        return x, final_hidden

class DotProductAttention(nn.Module):
    '''
    The paper did not specify an attention mechanism, so we implement a simple dot-product attention.
    '''
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        decoder_hidden: (batch, dec_dim)
        encoder_outputs: (batch, seq_len, enc_dim)
        mask: (batch, seq_len) bool (optional) where True indicates valid
        returns context vector: (batch, enc_dim) and attention weights (batch, seq_len)
        """
 
        # compute scores: batch x seq_len
        # decoder_hidden: (batch, enc_dim) -> (batch, enc_dim, 1)
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        # context = sum_t weights_t * encoder_outputs_t
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, enc_dim)
        return context, weights

class DecoderBlock(nn.Module):
    '''
    A single decoder block with attention mechanism.
    '''
    def __init__(self, hidden_size, num_layers, dropout, rnn_type, th, encoder_dim):
        super(DecoderBlock, self).__init__()
        self.th = th
        self.hidden_size = hidden_size
        self.encoder_dim = encoder_dim

        # Hidden State Mismatch
        # If Encoder (12) feeds into Decoder (128), need projection.
        if encoder_dim != hidden_size:
            self.hidden_proj = nn.Linear(encoder_dim, hidden_size)
            # Also need to project encoder outputs for Attention dot-product compatibility
            self.enc_out_proj = nn.Linear(encoder_dim, hidden_size)
            self.effective_enc_dim = hidden_size
        else:
            self.hidden_proj = None
            self.enc_out_proj = None
            self.effective_enc_dim = encoder_dim

        # RNN input is: previous_output (1) + context_vector (effective_enc_dim)
        rnn_input_dim = 1 + self.effective_enc_dim

        # RNN layer, 2 options
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTMCell(rnn_input_dim, hidden_size)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRUCell(rnn_input_dim, hidden_size)
        else:
            raise ValueError("RNN type must be GRU or LSTM")

        # map the rnn hidden output to scalar logit at each timestep
        self.out_linear = nn.Linear(hidden_size + self.effective_enc_dim, 1)
        self.attention = DotProductAttention()

    def forward(self, encoder_outputs, encoder_hidden, enc_mask=None):
        """
        encoder_outputs: (batch, seq_len, encoder_dim)  -- used for attention at each step
        encoder_hidden: encoder hidden (batch, dec_hidden_size)
        enc_mask: (batch, seq_len) optional mask for attention

        returns logits: (batch, th, 1)  (unsqueezed scalar per timestep)
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        # project Hidden State if dimensions mismatch
        h = encoder_hidden
        if self.hidden_proj is not None:
            h = self.hidden_proj(h)
            # Also project encoder outputs so Attention works (12 vs 128 dot product)
            encoder_outputs_proj = self.enc_out_proj(encoder_outputs)
        else:
            encoder_outputs_proj = encoder_outputs

        c = torch.zeros_like(h) # LSTM only

        # Initialize previous prediction (logit) to 0
        prev_out = torch.zeros(batch_size, 1).to(device)

        logits = []
        for t in range(self.th):
            # attention
            context, _ = self.attention(h, encoder_outputs_proj, enc_mask)
            
            rnn_input = torch.cat([prev_out, context], dim=1)
            
            if isinstance(self.rnn, nn.LSTMCell):
                h, c = self.rnn(rnn_input, (h, c))
            else:
                h = self.rnn(rnn_input, h)
            
            combined = torch.cat([h, context], dim=1)
            
            # ReLU decoder gave uniform distribution
            # logit = F.relu(self.out_linear(combined))
            
            logit = self.out_linear(combined)
            
            logits.append(logit)
            
            prev_out = logit
            
        return torch.stack(logits, dim=1)

class SurvivalSeq2Seq(nn.Module):
    """
    Full model:
        - encoder: GRU-D + optional stacked RNNs
        - attention: built inside DecoderBlock
        - K decoder blocks (one per event, but this is a one event implementation)
        - joint softmax across K * Th dimensions -> then reshape to (batch, K, Th)
    Inputs:
        grud_input: (batch, type_size=4, seq_len, feat) expected by GRUD
    """
    def __init__(self, input_size, hidden_size, num_layers, num_events, dropout, th, x_mean, rnn_type):
        super(SurvivalSeq2Seq, self).__init__()
        self.num_events = num_events
        self.th = th
        
        self.encoder = Encoder(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            rnn_type=rnn_type, 
            X_mean=x_mean
        )
        # If num_layers > 1, the stacked RNN projects to 'hidden_size' (128)
        # If num_layers == 1, the encoder outputs 'num_features' (12) due to GRUD restriction
        actual_encoder_dim = self.encoder.output_dim
        # decoder blocks for each hidden risk
        self.decoders = nn.ModuleList([
            DecoderBlock(
                hidden_size=hidden_size,
                num_layers=1,
                dropout=dropout,
                rnn_type=rnn_type,
                th=th,
                encoder_dim=actual_encoder_dim
            )
            for _ in range(num_events)
        ])
        # This is a 'sink' for probability, if user is censored assign high prob to this bin
        self.survival_logit = nn.Parameter(torch.zeros(1, 1))

    def forward(self, grud_input, enc_mask=None):
        """
        grud_input: (batch, 4, seq_len, feat) per GRUD's expectation
        enc_mask: optional (batch, seq_len) boolean mask for attention
        Returns:
          pdfs: (batch, K, Th) with softmax applied jointly across K*Th (so joint distribution sums to 1)
        """
        encoder_outputs, final_hidden = self.encoder(grud_input)  # (batch, seq_len, enc_dim)
        batch_size = encoder_outputs.size(0)

        # collect logits from each decoder: each gives (batch, th, 1)
        # this will only loop once since K=1 as per the paper
        decoder_outputs = []
        for k_idx, decoder in enumerate(self.decoders):
            logits_k = decoder(encoder_outputs, final_hidden, enc_mask)
            # flatten last dim
            #logits_k = logits_k.squeeze(-1)  # (batch, th)
            decoder_outputs.append(logits_k)

        # Concat events along the event dim
        stacked = torch.cat(decoder_outputs, dim=2) # (batch, th, num_events)
        
        # flatten to apply Joint Softmax
        flat = stacked.view(batch_size, -1) # (batch, num_events * th) this proj only does 1
        
        # append the survival logit
        survival_logit_expanded = self.survival_logit.expand(batch_size, 1)
        
        # Concat: (batch, num_events * th + 1)
        flat_extended = torch.cat([flat, survival_logit_expanded], dim=1)
        
        # ensures Sum(P(t)) + P(survival) = 1.0
        probs_extended = F.softmax(flat_extended, dim=1)
        
        # drop the survival bin
        # sum of 'probs' will now be <= 1.0-> resolves CDF(Tmax)=1 issue
        probs = probs_extended[:, :-1]
        
        # back to (batch, th, num_events)
        probs = probs.view(batch_size, self.th, self.num_events)
        
        # Permute to loss expected format: (batch, num_events, th)
        probs = probs.permute(0, 2, 1)
        
        return probs