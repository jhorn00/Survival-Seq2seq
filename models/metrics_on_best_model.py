import torch
import os
import pickle
from torch.utils.data import DataLoader

# Import modules
# Assumes this script is in the models/ directory along with other files
# Run with: python -m models.run_evaluation (from root)
from .MIMICIVDataset import MIMICIVDataset
from .utils import train_validate_test_split_parquet, collate_fn, evaluate
from .seq2seq_loss import SurvivalSeq2SeqLoss
from .eval_plots import (
    plot_figure_2, 
    generate_tables, 
    plot_ground_truth_vs_predicted, 
    plot_patient_example,
    plot_training_metrics,
    plot_patient_average_pdf
)

def metrics_on_best_model(
        output_path = "output/",
        data_path = "models/final_timeseries.parquet",
        batch_size = 256,
        alpha = 0.005,
        num_workers = 0
):
    # Config
    BATCH_SIZE = 512
    NUM_WORKERS = 0
    MODEL_PATH = os.path.join(output_path, "MyBestSeq2Seq.pth")
    SCALER_PATH = os.path.join(output_path, "scaler.pkl") # technically in dataset now
    METRICS_PATH = os.path.join(output_path, "best_metrics.pkl")
    DATASET_PATH = os.path.join(output_path, "test_dataset.pth") #TODO: change to valid for diff results
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Training History & Plot Curves
    if os.path.exists(METRICS_PATH):
        print("Loading training metrics...")
        with open(METRICS_PATH, "rb") as f:
            metrics_data = pickle.load(f)
        
        plot_training_metrics(metrics_data, output_dir=output_path)
    else:
        print(f"Warning: {METRICS_PATH} not found. Skipping loss curves.")

    # Load Scaler
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Did you run training?")
        
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")

    # Load datasets
    # weights_only=False is required because we are loading a custom class structure
    test_dataset = torch.load(DATASET_PATH, weights_only=False)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # Generate Evaluation Plots
    
    # Figure 2: Histograms
    print("Generating Figure 2...")
    plot_figure_2(test_dataset.labels_df, time_col='event_time_bin', event_col='event_flag', output_dir=output_path)

    # Tables 1 & 2: MAE/CI by Quantile
    print("Generating Tables...")
    generate_tables(model, test_loader, device)

    # Get Full Results for Figure 3
    criterion = SurvivalSeq2SeqLoss(alpha=alpha)
    print("Evaluating on Test Set...")
    _, _, _, results = evaluate(model, device, test_loader, criterion)
    
    # Figure 3: Ground Truth vs Predicted
    print("Generating Figure 3...")
    plot_ground_truth_vs_predicted(results, output_dir=output_path)

    # Figure 4: Patient Example
    print("Generating Figure 4...")
    plot_patient_example(model, test_loader, device, patient_idx=0, output_dir=output_path)

    # Figure 5: Average patient pdf
    print("Generating Figure 5...")
    plot_patient_average_pdf(model, test_loader, device, output_dir=output_path)

    print(f"All evaluation complete. Plots saved to current directory.")


    