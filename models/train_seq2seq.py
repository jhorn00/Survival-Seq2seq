import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from .seq2seq import SurvivalSeq2Seq
from .seq2seq_loss import SurvivalSeq2SeqLoss

from .MIMICIVDataset import MIMICIVDataset
from .utils import train, evaluate, train_validate_test_split_parquet, collate_fn

from .eval_plots import plot_figure_2, generate_tables, plot_ground_truth_vs_predicted, plot_patient_example, plot_patient_average_pdf, plot_training_metrics
# Note: Code taken from HW4 and modified for seq2seq model training


def plot_from_model(model, test_dataset, test_loader, device, criterion, metrics_data, output_dir="./"):
	# FIGURE 2 (Histogram)
	plot_figure_2(test_dataset.labels_df, time_col='event_time_bin', event_col='event_flag', output_dir=output_dir)

	# TABLES 1 & 2
	metrics = generate_tables(model, test_loader, device) 
	# metrics dictionary now contains {'25%': {'MAE': ..., 'CI': ...}, ...}

	# FIGURE 3 (Pred vs Truth)
	# need 'results' list from your evaluate() function
	_, _, _, results = evaluate(model, device, test_loader, criterion)
	plot_ground_truth_vs_predicted(results, output_dir=output_dir)

	# FIGURE 4 (Single Patient PDF)
	# TODO: Find patient that is uncensored
	plot_patient_example(model, test_loader, device, patient_idx=0, output_dir=output_dir)

	plot_patient_average_pdf(model, test_loader, device, output_dir=output_dir)

	plot_training_metrics(metrics=metrics_data, output_dir=output_dir)


def train_seq2seq(
		output_path = "output/",
		data_path = "models/final_timeseries.parquet", # We just put it in the models dir since the other dirs are cluttered
		num_epochs = 50,
		batch_size = 256,
		alpha = 0.005,
		use_cuda = True, # Set 'True' if you want to use GPU
		num_workers = 0,
		reduce_dataset_ratio = 0.10 # fraction of dataset to use for testing - SET THIS TO 1 FOR PAPER RESULTS
):
	os.makedirs(output_path, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

	torch.manual_seed(1)
	if device.type == "cuda":
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	# Data loading
	print('===> Loading dataset')

	train_df, validate_df, test_df = train_validate_test_split_parquet(data_path, validate_size_full=0.2, test_size=0.1, random_state=5, group_fraction=reduce_dataset_ratio)
	train_dataset = MIMICIVDataset(train_df, is_train=True)
	print(f"Number of training rows: {len(train_dataset.df)}")
	print(f"Number of training hospital stays: {len(train_dataset.df["stay_id"].unique())}")
	# Pass scaler in order to use the same scaling as for training set
	valid_dataset = MIMICIVDataset(validate_df, is_train=False, scaler=train_dataset.scaler)
	test_dataset = MIMICIVDataset(test_df, is_train=False, scaler=train_dataset.scaler)

	train_loader = DataLoader(
			train_dataset, 
			batch_size=batch_size, 
			shuffle=True, 
			collate_fn=collate_fn,
			num_workers=num_workers,
			pin_memory=True
		)
		
	valid_loader = DataLoader(
		valid_dataset, 
		batch_size=batch_size, 
		shuffle=False, 
		collate_fn=collate_fn,
		num_workers=num_workers,
		pin_memory=True
	)

	test_loader = DataLoader(
		test_dataset, 
		batch_size=batch_size, 
		shuffle=False, 
		collate_fn=collate_fn,
		num_workers=num_workers,
		pin_memory=True
	)

	# Calculate empirical means from the training dataframe
	# TODO: Does order match features !!!!!!!!!!!!!
	# Assuming train_dataset.feature_cols matches the model input order:
	X_mean_list = train_df[train_dataset.feature_cols].mean().tolist()
	X_mean_tensor = torch.tensor(X_mean_list, dtype=torch.float32).to(device)

	real_input_size = len(train_dataset.feature_cols)

	model = SurvivalSeq2Seq(
			input_size=real_input_size,
			hidden_size=120,          # for both Encoder and Decoder, paper used 120
			num_layers=1,             # set to 2 Layers if you want a second of either gru or lstm
			num_events=1,             # mortality only
			dropout=0.4,			  # Paper uses 0.4 dropout
			th=198,                   # Time Horizon (bins)
			x_mean=X_mean_tensor,     
			rnn_type="GRU"
		)
	# Log-Likelihood + Ranking Loss
	criterion = SurvivalSeq2SeqLoss(alpha=alpha)
	optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0) # TODO: weight decay value?

	model.to(device)
	criterion.to(device)

	# Save scaler for current run
	with open(os.path.join(output_path, "scaler.pkl"), "wb") as f:
		pickle.dump(train_dataset.scaler, f)

	best_val_mae = float('inf')

	train_losses, train_maes, train_cis = [], [], []
	valid_losses, valid_maes, valid_cis = [], [], []

	epoch_times = []
	for epoch in range(num_epochs):
		epoch_timer = time.time()

		train_loss, train_mae, train_ci = train(model, device, train_loader, criterion, optimizer, epoch)
		valid_loss, valid_mae, valid_ci, valid_results = evaluate(model, device, valid_loader, criterion)

		train_losses.append(train_loss)
		valid_losses.append(valid_loss)

		train_maes.append(train_mae)
		valid_maes.append(valid_mae)

		train_cis.append(train_ci)
		valid_cis.append(valid_ci)

		# Model with lowest validation MAE is best: best at predicting sepcific time bin
		# Can also consider CI which would find model best at ranking patients time to event
		is_best = valid_mae < best_val_mae
		if is_best:
			print(f"NEW BEST MODEL: MAE improved from {best_val_mae:.3f} to {valid_mae:.3f}")
			best_val_mae = valid_mae
			torch.save(model, os.path.join(output_path, "MyBestSeq2Seq.pth"), _use_new_zipfile_serialization=False)
		delta = time.time() - epoch_timer
		print(f'Epoch {epoch+1} complete in {delta:.2f}s')
		epoch_times.append(delta)

	average_epoch_seconds = np.mean(epoch_times)
	total_train_seconds = np.sum(epoch_times)
	print(f"Total training time: {total_train_seconds:.2f}s\nAverage epoch time: {average_epoch_seconds:.2f}s")

	# Save off metrics
	metrics_data = {
		"train_losses": train_losses, "valid_losses": valid_losses,
		"train_maes": train_maes, "valid_maes": valid_maes,
		"train_cis": train_cis, "valid_cis": valid_cis,
		"average_epoch_seconds": average_epoch_seconds, "total_train_seconds": total_train_seconds
	}
	with open(os.path.join(output_path, "best_metrics.pkl"), "wb") as f:
		pickle.dump(metrics_data, f)

	# Save final model
	torch.save(model, os.path.join(output_path, "FinalSeq2Seq.pth"), _use_new_zipfile_serialization=False)
	print("Saving Test Dataset...")
    # save the dataset object, which contains the pre-split/scaled DataFrame
	torch.save(test_dataset, os.path.join(output_path, "test_dataset.pth"))
	torch.save(valid_dataset, os.path.join(output_path, "valid_dataset.pth"))
	# Plots
	print("===> Generating Plots for Best Model...") #TODO: save last model or best model or both?
	best_model = torch.load(os.path.join(output_path, "MyBestSeq2Seq.pth"), weights_only=False)
	best_model.to(device)
	plot_from_model(best_model, test_dataset, test_loader, device, criterion, metrics_data=metrics_data, output_dir=output_path)

print("Training complete.")
