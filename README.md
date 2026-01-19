# Survival Seq2seq
Reproducing the results of Survival Seq2Seq: A Survival Model based on Sequence to Sequence Architecture

## Getting Started

### Dataset
**Important:** You MUST obtain a copy of MIMIC-IV 1.0 on your own to run the preprocessing or training code. We would be in violation of data access agreements if we were to share any of the MIMIC data ourselves. We have provided a few trained models for your personal use if you don't wish to train them yourself.  

If you would prefer to follow the data-access directions on MIT.edu, you can find them [here](https://mimic.mit.edu/docs/gettingstarted/).

- Physionet provides the steps you must complete for data access in the ["Files"](https://physionet.org/content/mimiciv/1.0/#files) section of the MIMIC-IV 1.0 dataset.
- Once you are properly credentialed, download/access options will appear.
    - As of January 2026, you have the options below. We recommend the direct download with `wget` since the total uncompressed size is only 6.9 GB.
        - `wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/mimiciv/1.0/`
        - ZIP archive (6.9 GB, same as the uncompressed)
        - AWS Command Line
        - Google BigQuery
    - After dowloading the dataset, plese validate the files (`sha256sum -c SHA256SUMS.txt`)
- At this point, you will want to take note of the directory your MIMIC files are in and head over to the `preprocessing/` directory.

---

### Environment
- Make sure you have [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) installed. Alternatively, you may wish to follow along with the package list in `environment.yml` to create your own local Python environment/venv.
- Verify installation with `conda --version`
- Create the environment from the .yml file with `conda env create -f environment.yml`
- Activate the environment with `conda activate survival`

---

### Repo layout
Every directory provides a README file, should you require more information than is provided here.
- `models/`
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

---

### Models
Models can be trained or run using the top-level python scripts under models. Further model details can be found in the README under `models/`.

---

## References
- Data source
    - [MIMIC-IV 1.0](https://physionet.org/content/mimiciv/1.0/)
- Data processing
    - [mimic-code](https://github.com/MIT-LCP/mimic-code) was extremely helpful in determining what metrics we should consider for patient time-series data.
- Helper Libraries
    - [GRU-D](https://github.com/zhiyongc/GRU-D): GRU-D implementation based on "Recurrent neural networks for multivariate time series with missing values" by Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan (2018).
