# survival-seq2seq
Reproducing the results of Survival Seq2Seq: A Survival Model based on Sequence to Sequence Architecture

Team 26, Team ID: F1
- Dawson Horn
- Michael Falter

## Getting Started
Repo layout:
- `models/`
    - Check directory README for more details.
    - Contains model exploration, training, and testing files.
    - Contains final preprocessed data parquet after data pipeline has been run.
- `preprocessing/`
    - `pipeline.py` in this directory must be run before anything else will work!
    - Check directory README for more details.
    - Contains MIMIC data preprocessing files.
- `environment.yml`
    - Conda env file for running code in this repo.
- `main.py`
    - This was used by our team to generate the metrics and information found in the report.
    - This file assumes you have already run preprocessing.
    - This is the entrypoint for most testing. There are constants for configuring whether you want to run Seq2seq or DDH, as well as a modeling function. Select desired hyperparameters if you wish to test other configurations.
- `survival_seq2seq_grid.py`
    - This was used by our team to run grid search to test different hyperparameters.

### Dataset
Once you have obtained the MIMIC-IV 1.0 dataset, refer to the README in `preprocessing/` for data processing directions.

#### [Optional] Cloud
- [Link your cloud account to get data access if you want](https://mimic.mit.edu/docs/gettingstarted/cloud/link/)
    - Apparently for the data we want it is about 7GB so not bad at all.

#### Data Access (Same for local and cloud)
- Access cloud data:
    - [Data page you will want](https://physionet.org/content/mimiciv/1.0/)
        - You will need to sign the access agreement for the dataset we want (you just click a button since we have the training). Ctrl+f for 'files' and you should see it at the bottom of the page.
        - The command to download is `wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/mimiciv/1.0/`. There are also ZIP and Google BigQuery options.
        - Validate files `sha256sum -c SHA256SUMS.txt`
    - [Full instructions if you want](https://mimic.mit.edu/docs/gettingstarted/cloud/request/)


### Conda Environment
- Install from `environment.yml` with `conda env create -f environment.yml`

### Models
Models can be trained or run using the top-level python scripts under models. Further model details can be found in the README under `models/`.

## References
- Data source
    - [MIMIC-IV 1.0](https://physionet.org/content/mimiciv/1.0/)
- Data processing
    - [mimic-code](https://github.com/MIT-LCP/mimic-code) was extremely helpful in determining what metrics we should consider for patient time-series data.
- Helper Libraries
    - [Dynamic Deep Hit](https://github.com/Jeanselme/DynamicDeepHit)
    - [Deep Survival Machines](https://github.com/autonlab/DeepSurvivalMachines/tree/c454774199c389e7bb9fa3077f153cdf4f1e7696)
    - [GRU-D](https://github.com/zhiyongc/GRU-D)