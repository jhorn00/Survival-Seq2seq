from models.train_seq2seq import train_seq2seq
from models.metrics_on_best_model import metrics_on_best_model

print("Running model training...")
RUN_METRICS = True
OUTPUT_PATH = "full_run/"
DATA_PATH = "models/final_timeseries.parquet"

print("Training Surivial Seq2Seq...")
train_seq2seq(
    output_path=OUTPUT_PATH,
    data_path=DATA_PATH,
    num_epochs=5, # make 50 for final run
    batch_size=512,
    alpha=0.1,
    use_cuda=True,
    num_workers=0,
    reduce_dataset_ratio=1.0 # make 1 for final run
)

if RUN_METRICS:
    print("Running metrics...")
    metrics_on_best_model(
        output_path=OUTPUT_PATH,
        data_path=DATA_PATH,
        batch_size=512,
        alpha=0.1,
        num_workers=0
    )
