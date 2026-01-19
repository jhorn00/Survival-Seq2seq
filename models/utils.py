import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from typing import Tuple

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def collate_fn(batch):
    """
    Pads batch of variable length sequences.
    """
    max_batch_len = max([b['length'] for b in batch])
    num_features = batch[0]['X'].shape[1]
    
    batch_size = len(batch)
    
    # 4 components for GRU-D input
    X_padded = torch.zeros(batch_size, max_batch_len, num_features)
    X_last_padded = torch.zeros(batch_size, max_batch_len, num_features)
    Mask_padded = torch.zeros(batch_size, max_batch_len, num_features)
    Delta_padded = torch.zeros(batch_size, max_batch_len, num_features)
    
    Labels = torch.zeros(batch_size, 2)
    Lengths = torch.zeros(batch_size)

    for i, b in enumerate(batch):
        l = b['length']
        # Fill tensors
        X_padded[i, :l, :] = torch.from_numpy(b['X'])
        X_last_padded[i, :l, :] = torch.from_numpy(b['X_last'])
        Mask_padded[i, :l, :] = torch.from_numpy(b['Mask'])
        Delta_padded[i, :l, :] = torch.from_numpy(b['Delta'])
        
        Labels[i] = torch.from_numpy(b['y'])
        Lengths[i] = l
        
    # Stack for GRUD: (Batch, Type_Size=4, Seq_Len, Features)
    # 0:X, 1:X_last, 2:Mask, 3:Delta
    grud_input = torch.stack([X_padded, X_last_padded, Mask_padded, Delta_padded], dim=1)
    
    return grud_input, Labels

def grouped_train_test_split_parquet(data_path: str, group_col = "stay_id", test_size = 0.1, group_fraction = 1.0, random_state = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
	data_df = pd.read_parquet(data_path)

	groups = data_df[group_col].unique()
	rng = np.random.RandomState(random_state)

	# Subsampling the data to speed up development
	if group_fraction < 1.0:
		n_keep = max(1, int(len(groups) * group_fraction))
		groups = rng.choice(groups, size=n_keep, replace=False)

	train_groups, test_groups = train_test_split(groups, test_size=test_size, random_state=random_state)

	train_df = data_df[data_df[group_col].isin(train_groups)].reset_index(drop=True)
	test_df = data_df[data_df[group_col].isin(test_groups)].reset_index(drop=True)

	return train_df, test_df

def train_validate_test_split_parquet(data_path: str, group_col = "stay_id", validate_size_full=0.1, test_size = 0.1, group_fraction = 1.0, random_state = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	data_df = pd.read_parquet(data_path)

	groups = data_df[group_col].unique()
	rng = np.random.RandomState(random_state)

	# Subsampling the data to speed up development
	if group_fraction < 1.0:
		n_keep = max(1, int(len(groups) * group_fraction))
		groups = rng.choice(groups, size=n_keep, replace=False)

	train_groups, test_groups = train_test_split(groups, test_size=test_size, random_state=random_state)

	validate_size = float(validate_size_full / (1 - test_size))

	train_groups, validate_groups = train_test_split(train_groups, test_size=validate_size, random_state=random_state)

	train_df = data_df[data_df[group_col].isin(train_groups)].reset_index(drop=True)
	validate_df = data_df[data_df[group_col].isin(validate_groups)].reset_index(drop=True)
	test_df = data_df[data_df[group_col].isin(test_groups)].reset_index(drop=True)


	return train_df, validate_df, test_df


def get_expected_time(pdf, time_resolution=1.0):
	"""
    Calculates the expected time from the discrete PDF, see equation 2 of paper
	This produces the CDF-based expected time as described in the paper
    pdf: (batch_size, num_time_bins) - output after softmax
    time_resolution: scalar representing the real-world time of one bin (e.g., 4 hours)
    """
	
	# add normalize to force the prob to equal 1 (account for the sink timeblock we added for survivor predictions)
	sum_probs = pdf.sum(dim=1, keepdim=True)
	normalized_pdf = pdf / (sum_probs + 1e-8)

	batch_size, num_bins = pdf.shape
	# Time vector (0, 1, 2, ..., num_bins-1)
	time_indices = torch.arange(0, num_bins, device=pdf.device).float()
	# Expected value = Sum(t * p(t))
	expected_time_bins = torch.sum(normalized_pdf * time_indices, dim=1)
	# expected_time_bins = torch.sum(pdf * time_indices, dim=1)
	# Convert to real-world time
	expected_time = expected_time_bins * time_resolution
	return expected_time

def compute_mae(predicted_pdfs, labels, time_res=1.0):
	"""
	predicted_pdfs: (batch, num_bins)
    true_times: (batch, ) - Ground truth time index
    true_events: (batch, ) - 1 if event occurred (uncensored), 0 if censored
	"""
	true_times = labels[:, 1]
	true_events = labels[:, 0]
	# Get predicted epxected times
	pred_times = get_expected_time(predicted_pdfs, time_resolution=time_res)  # (batch, )
	# Filter for uncensored only
	uncensored_mask = (true_events == 1)
	if torch.sum(uncensored_mask) == 0:
		return 0.0 # Avoids div by zero
	
	valid_preds = pred_times[uncensored_mask]
	valid_true_labels = true_times[uncensored_mask] #ToDo: is this in hours or bin count?

	# MAE
	mae = torch.mean(torch.abs(valid_preds - valid_true_labels))
	return mae.item()

def compute_time_dependent_ci(predicted_pdfs, labels):
    """
    Computes Time-Dependent Concordance Index (CI) as defined in the paper as:
    CI(t) = P(F(t|xi) > F(t|xj) | delta_i=1, Ti < Tj, Ti < t)
    
    Evaluates if the model correctly assigns a higher risk (CDF) 
    to patient who experienced the event sooner (patient i) compared to 
    a patient who survived longer (patient j) at the time of i's event.
    """
    # CDF from PDF
    # predicted_pdfs shape: (Batch, TimeBins)
    cdfs = torch.cumsum(predicted_pdfs, dim=1)
    
    death_events = labels[:, 0].bool() # 1 if event, 0 if censored
    death_times = labels[:, 1].long()  # Time bin index for death or censor
    
    concordant_pairs = 0
    total_pairs = 0
    
    # Iterate over all samples to find valid pairs
    # O(N^2)
    # Could vectorize, may need to
    batch_size = len(labels)

    for i in range(batch_size):
        # Patient i must be uncensored to be the "first" in the pair
        if not death_events[i]:
            continue
            
        death_time_i = death_times[i]
        
        # Determine risk of patient i at their event time
        # Clamp death_time_i within prediction horizon
        if death_time_i >= cdfs.shape[1]:
             death_time_i = cdfs.shape[1] - 1
             
        risk_i = cdfs[i, death_time_i]
        
        for j in range(batch_size):
            if i == j: 
                continue
            
            death_time_j = death_times[j]
            
            # Valid Pair: T_i < T_j
            # (Patient i had event before Patient j had event or was censored)
            if death_time_i < death_time_j:
                total_pairs += 1
                
                # Risk at time t = T_i
                # Does Patient i have higher cumulative risk at time T_i than Patient j?
                risk_j = cdfs[j, death_time_i]
                
                if risk_i > risk_j:
                    concordant_pairs += 1
                elif risk_i == risk_j:
                    concordant_pairs += 0.5 # Tie
                    
    if total_pairs == 0:
        return 0.0
    # If value is 1.0, perfect concordance (Model always assigns higher risk to those who die sooner), lower is worse
    return concordant_pairs / total_pairs

def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	mae_meter = AverageMeter()
	ci_meter = AverageMeter()

	model.train()

	end = time.time()
	for i, (input, target) in enumerate(data_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		output = model(input)
		# Convert output to shape (batch, num_classes) from (batch, 1, num_classes)
		pdf_output = output.squeeze(dim=1)
		
		# TODO: Does loss expect logits or probs?
		loss = criterion(pdf_output, target)#criterion(output, target)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()


		cur_batch_size = target.size(0)
		losses.update(loss.item(), cur_batch_size)
		# Metrics
		mae_val = compute_mae(pdf_output, target)
		# prevent from collecting a ton of 0 MAE values and skewing our MAE by a lot (in a good direction)
		num_uncensored = torch.sum(target[:, 0] == 1).item()
		if num_uncensored > 0:
			mae_meter.update(mae_val, num_uncensored)
		ci_val = compute_time_dependent_ci(pdf_output, target)
		ci_meter.update(ci_val, cur_batch_size)

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae.val:.3f} ({mae.avg:.3f})\t'
                  'CI {ci.val:.3f} ({ci.avg:.3f})'.format(
                epoch, i, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mae=mae_meter, ci=ci_meter))

	return losses.avg, mae_meter.avg, ci_meter.avg


def evaluate(model, device, data_loader, criterion, print_freq=10):
	batch_time = AverageMeter()
	losses = AverageMeter()
	mae_meter = AverageMeter()
	ci_meter = AverageMeter()

	results = []

	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input)
			pdf_output = output.squeeze(dim=1)
			
			loss = criterion(pdf_output, target)#criterion(output, target) # TODO: Does loss expect logits or probs?
			# Metrics
			mae_val = compute_mae(pdf_output, target)
			ci_val = compute_time_dependent_ci(pdf_output, target)
			
			cur_batch_size = target.size(0)
			losses.update(loss.item(), cur_batch_size)
			# prevent from collecting a ton of 0 MAE values and skewing our MAE by a lot (in a good direction)
			num_uncensored = torch.sum(target[:, 0] == 1).item()
			if num_uncensored > 0:
				mae_meter.update(mae_val, num_uncensored)
			ci_meter.update(ci_val, cur_batch_size)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			# Storing predicted expected time vs true time
			y_true = target.detach().cpu().numpy()
			y_pred_time = get_expected_time(pdf_output).detach().cpu().numpy()
			# Store tuples of (TrueEvent, TrueTime, PredTime)
			# target[:, 0] is event, target[:, 1] is time
			batch_res = list(zip(y_true[:,0], y_true[:,1], y_pred_time))
			results.extend(batch_res)

			if i % print_freq == 0:
				print('Evaluate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae.val:.3f} ({mae.avg:.3f})\t'
                      'CI {ci.val:.3f} ({ci.avg:.3f})'.format(
                    i, len(data_loader), batch_time=batch_time, 
                    loss=losses, mae=mae_meter, ci=ci_meter))

	return losses.avg, mae_meter.avg, ci_meter.avg, results
