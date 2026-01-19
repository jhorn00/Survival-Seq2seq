from models.train_seq2seq import train_seq2seq
from models.metrics_on_best_model import metrics_on_best_model

BASE_OUTPUT_PATH = "output/"
DATA_PATH = "models/final_timeseries.parquet"
USE_CUDA = True
NUM_WORKERS = 0

num_epochs_list = [1, 5, 10]
batch_size_list = [128, 256, 512]
alpha_list = [0.01, 0.001, 0.0001]
reduce_dataset_ratio_list = [0.1]

for num_epochs in num_epochs_list:
    for batch_size in batch_size_list:
        for alpha in alpha_list:
            for dataset_ratio in reduce_dataset_ratio_list:
                print("====================")
                print(f"Running combination:\nnum_epochs: {num_epochs}\nbatch_size: {batch_size}\nalpha: {alpha}\npercent of full dataset for training: {int(dataset_ratio * 100)}%")
                print("--------------------")

                output_path = BASE_OUTPUT_PATH + f"ne{num_epochs}_b{batch_size}_a{alpha}_dpct{int(dataset_ratio * 100)}/"
                train_seq2seq(
                    output_path=output_path,
                    data_path=DATA_PATH,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    alpha=alpha,
                    use_cuda=USE_CUDA,
                    num_workers=NUM_WORKERS,
                    reduce_dataset_ratio=dataset_ratio
                )

                metrics_on_best_model(
                    output_path=output_path,
                    data_path=DATA_PATH,
                    batch_size=batch_size,
                    alpha=alpha,
                    num_workers=NUM_WORKERS
                )

                print("====================")
