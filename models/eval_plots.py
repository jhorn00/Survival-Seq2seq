import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from .utils import compute_mae, compute_time_dependent_ci, get_expected_time

def plot_training_metrics(metrics, output_dir = "./"):
    """
    Plots Loss, MAE, and CI over epochs from the metrics dictionary.
    """
    # Check if metrics are populated
    if not metrics or 'train_losses' not in metrics:
        print("No training metrics found to plot.")
        return

    epochs = range(1, len(metrics['train_losses']) + 1)
    
    # Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['train_losses'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, metrics['valid_losses'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir + "training_loss_curve.png")
    plt.close()
    print("Saved training_loss_curve.png")

    # Dual MAE and CI
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # MAE Subplot
    ax1.plot(epochs, metrics['train_maes'], 'b-', label='Train MAE')
    ax1.plot(epochs, metrics['valid_maes'], 'r-', label='Valid MAE')
    ax1.set_title('Mean Absolute Error (MAE)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MAE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CI Subplot
    ax2.plot(epochs, metrics['train_cis'], 'b-', label='Train CI')
    ax2.plot(epochs, metrics['valid_cis'], 'r-', label='Valid CI')
    ax2.set_title('Concordance Index (CI)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('CI (Higher is better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir + "training_metrics_curves.png")
    plt.close()
    print("Saved training_metrics_curves.png")


def plot_figure_2(df, time_col='event_time_bin', event_col='event_flag', label_map={0: 'Censored', 1: 'Event'}, output_dir = "./"):

    """
    Implements Figure 2: Histogram of survival times.
    
    Args:
        df: DataFrame containing the labels and times.
        time_col: Column name for event time (bins or actual time).
        event_col: Column name for event indicator.
        label_map: Dictionary mapping event codes to names (e.g., {0: 'Censored', 1: 'Death'}).
    """
    plt.figure(figsize=(10, 6))
    
    # Iterate through each event type to plot its histogram/KDE
    unique_events = sorted(df[event_col].unique())
    
    for event_code in unique_events:
        subset = df[df[event_col] == event_code]
        label = label_map.get(event_code, f'Event {event_code}')
        
        # Smooth KDE + Histogram
        sns.histplot(
            subset[time_col], 
            label=label, 
            kde=True, 
            element="step", 
            alpha=0.3,
            line_kws={'linewidth': 2}
        )

    plt.xlabel("Survival Time (time index)")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Histogram of Survival Times")

    plt.savefig(output_dir + "figure_2_survival_histogram.png")
    #plt.show()

    
def generate_tables(model, dataloader, device):
    """
    Generates data for Tables 1 and 2: MAE and CI by Quantiles.
    
    Full pass over the dataloader to collect all PDFs and Targets
    """
    model.eval()
    all_pdfs = []
    all_targets = []

    print("Collecting predictions for tables...")
    with torch.no_grad():
        for i, (input_data, target) in enumerate(dataloader):
            if isinstance(input_data, tuple):
                input_data = tuple([e.to(device) if isinstance(e, torch.Tensor) else e for e in input_data])
            else:
                input_data = input_data.to(device)
            target = target.to(device)

            output = model(input_data)
            output = output.squeeze(dim=1)
            # double softmax again
            # pdf_output = torch.softmax(output, dim=1)
            pdf_output = output

            all_pdfs.append(pdf_output.cpu())
            all_targets.append(target.cpu())

    # Concatenate all batches
    full_pdfs = torch.cat(all_pdfs, dim=0)   # (N, TimeBins)
    full_targets = torch.cat(all_targets, dim=0) # (N, 2)

    # Sort by Ground Truth Time (target[:, 1]) to create quantiles
    true_times = full_targets[:, 1]
    sorted_indices = torch.argsort(true_times)
    
    sorted_pdfs = full_pdfs[sorted_indices]
    sorted_targets = full_targets[sorted_indices]
    
    N = len(sorted_targets)
    quantiles = [0.25, 0.50, 0.75, 1.0]
    results = {}
    print(f"{'-'*40}\n")
    print(f"{'Quantile':<10} | {'MAE':<10} | {'CI':<10}")
    print(f"{'-'*40}\n")
    for q in quantiles:
        cutoff = int(N * q)
        if cutoff == 0: continue

        # Slice the top q% of the data
        subset_pdfs = sorted_pdfs[:cutoff]
        subset_targets = sorted_targets[:cutoff]

        # Compute Metrics
        mae = compute_mae(subset_pdfs, subset_targets)
        
        # CI calculation
        ci = compute_time_dependent_ci(subset_pdfs, subset_targets)

        results[f"{int(q*100)}%"] = {'MAE': mae, 'CI': ci}
        
        print(f"{int(q*100)}%       | {mae:.3f}      | {ci:.3f}")

    print(f"{'-'*40}\n")
    return results

# Plots.py wrappers
def plot_ground_truth_vs_predicted(results_list, sample_limit=70, output_dir = "./"):
    """
    Figure 3. Moving from plots.py
    Args:
        results_list: List of tuples (TrueEvent, TrueTime, PredTime) from evaluate()
    """
    # results_list contains: (y_true[:,0], y_true[:,1], y_pred_time)
    event_indicators, true_times, pred_times = zip(*results_list)
    
    # Convert to lists for the original function logic
    # Filter for uncensored only (event == 1)
    uncensored_pairs = []
    for e, t, p in zip(event_indicators, true_times, pred_times):
        if e == 1:
            uncensored_pairs.append((t, p))
            
    # Sort by True Time
    # uncensored_pairs.sort(key=lambda x: x[0])
    
    if not uncensored_pairs:
        print("No uncensored data to plot.")
        return

    # Limit samples
    if len(uncensored_pairs) > sample_limit:
        uncensored_pairs = uncensored_pairs[:sample_limit]
        
    y_true, y_pred = zip(*uncensored_pairs)

    plt.figure(figsize=(12, 6))
    x = range(len(y_pred))

    plt.step(x, y_true, where='mid', label='Ground Truth', linestyle='-', color='blue', linewidth=1.5)
    plt.step(x, y_pred, where='mid', label='Predicted Time', linestyle='--', color='orange', linewidth=1.5)
    plt.xlabel("Uncensored Samples (Sorted by Length)")
    plt.ylabel("Time of Event")
    plt.legend()
    plt.title("Figure 3: Ground Truth vs Predicted Times (Uncensored)")
    plt.grid(True, alpha=0.3)

    plt.savefig(output_dir + "figure_3_predictions.png")
    #plt.show()

    
def plot_patient_example(model, dataloader, device, patient_idx=0, output_dir = "./"):
    """
    Figure 4.
    """
    model.eval()
    
    # Get a single batch
    input_data, target = next(iter(dataloader))
    
    if isinstance(input_data, tuple):
        input_data = tuple([e.to(device) if isinstance(e, torch.Tensor) else e for e in input_data])
    else:
        input_data = input_data.to(device)
    
    with torch.no_grad():
        output = model(input_data)
        output = output.squeeze(dim=1)
        pdf_output = output # removed softmax because model does this
        
        # Calculate expected time
        pred_times = get_expected_time(pdf_output) # (Batch,)
    
    for i in range(len(pdf_output)):
        if target[i, 0].item() == 1:
            patient_idx = i
            break

    # Extract data for specific patient
    pdf = pdf_output[patient_idx].cpu().numpy()
    pred_time = pred_times[patient_idx].item()
    true_event = target[patient_idx, 0].item()
    true_time = target[patient_idx, 1].item()
    
    title = f"Patient PDF (Status: {'Uncensored' if true_event==1 else 'Censored'})"
    
    plt.figure(figsize=(10, 6))
    plt.plot(pdf, label='Predicted PDF', color='black')

    if true_event == 1:
        plt.axvline(x=true_time, color='green', linestyle='-', linewidth=2, label=f'Ground Truth ({true_time:.1f})')
        plt.axvline(x=pred_time, color='red', linestyle='--', linewidth=2, label=f'Predicted ({pred_time:.1f})')
    else:
        plt.axvline(x=true_time, color='gray', linestyle=':', linewidth=2, label=f'Censored Time ({true_time:.1f})')
        plt.axvline(x=pred_time, color='red', linestyle='--', linewidth=2, label=f'Predicted ({pred_time:.1f})')

    plt.xlabel('Prediction Horizon (Time Bins)')
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend()
    plt.savefig(output_dir + "figure_4_patient_pdf.png")
    #plt.show()


def plot_patient_average_pdf(model, dataloader, device, output_dir = "./"):
    """
    Like plot_patient_example, but plots an average PDF over many patients.

    - Uses a single batch from the dataloader (same as original).
    - If there are uncensored patients in the batch, averages over those.
      Otherwise, averages over all patients in the batch.
    """
    model.eval()
    
    # Get a single batch (same as original)
    input_data, target = next(iter(dataloader))
    
    if isinstance(input_data, tuple):
        input_data = tuple([e.to(device) if isinstance(e, torch.Tensor) else e for e in input_data])
    else:
        input_data = input_data.to(device)
    
    target = target.to(device)

    with torch.no_grad():
        output = model(input_data)
        output = output.squeeze(dim=1)               # (Batch, Bins)
        pdf_output = output # rem softmax
        
        # Calculate expected time per patient
        pred_times = get_expected_time(pdf_output)   # (Batch,)

    # event status and times from target
    events = target[:, 0]  # 1 = uncensored, 0 = censored
    times  = target[:, 1]

    # Prefer averaging over uncensored patients if any exist
    uncensored_mask = (events == 1)
    if uncensored_mask.any():
        mask = uncensored_mask
        status_str = "Uncensored Only"
    else:
        mask = torch.ones_like(events, dtype=torch.bool)
        status_str = "All Patients (no uncensored in batch)"

    # Apply mask
    pdf_sel       = pdf_output[mask]      # (N, Bins)
    pred_times_sel = pred_times[mask]     # (N,)
    times_sel      = times[mask]          # (N,)
    events_sel     = events[mask]         # (N,)

    # Compute averages
    avg_pdf = pdf_sel.mean(dim=0).cpu().numpy()
    mean_true_time = times_sel.float().mean().item()
    mean_pred_time = pred_times_sel.float().mean().item()
    event_rate = events_sel.float().mean().item()  # fraction uncensored in selection

    title = f"Average Patient PDF ({status_str}, N={pdf_sel.size(0)} patients, event_rate={event_rate:.2f})"
    
    plt.figure(figsize=(10, 6))
    plt.plot(avg_pdf, label='Average Predicted PDF', color='black')

    # Mean true and predicted times as vertical lines
    plt.axvline(x=mean_true_time, color='green', linestyle='-',  linewidth=2,
                label=f'Mean Ground Truth ({mean_true_time:.1f})')
    plt.axvline(x=mean_pred_time, color='red',   linestyle='--', linewidth=2,
                label=f'Mean Predicted ({mean_pred_time:.1f})')

    plt.xlabel('Prediction Horizon (Time Bins)')
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir + "figure_5_patient_average_pdf.png")
    # plt.show()
