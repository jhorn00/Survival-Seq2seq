# This file will make sure the local data is configured properly for preprocessing steps.

import os

# Local imports
from mimic_pipeline.duck import configure_duckdb
from mimic_pipeline.io_registry import csv_to_parquet

def main():
    base_path = "/mnt/d/mimic-4-1.0/physionet.org/files/mimiciv/1.0/"
    directory_map = {
        # core
        "ADMISSIONS": "core/admissions.csv.gz",
        "PATIENTS": "core/patients.csv.gz",

        # hospital
        "DIAGNOSES_ICD": "hosp/diagnoses_icd.csv.gz",
        "D_ICD_DIAGNOSES": "hosp/d_icd_diagnoses.csv.gz",

        # icu
        "ICUSTAYS": "icu/icustays.csv.gz",
        "INPUTEVENTS": "icu/inputevents.csv.gz",
        "OUTPUTEVENTS": "icu/outputevents.csv.gz",
        "CHARTEVENTS": "icu/chartevents.csv.gz",
        "D_ITEMS": "icu/d_items.csv.gz",
    }
    filepath_map = {
        key: base_path + value for key, value in directory_map.items()
    }


    data_dir = "data/"
    os.makedirs(data_dir, exist_ok=True)

    ddb = configure_duckdb(database_name="data/data.duckdb", memory_limit="8GB", temp_directory=base_path)
    csv_to_parquet(ddb, filepath_map, out_dir="data/base_parquet/")

if __name__ == "__main__":
    main()
