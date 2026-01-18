# Preprocessing script for MIMIC-IV 1.0 data

####################################################################################################
############### THIS IS WHAT YOU SHOULD CHANGE ACCORDING TO YOUR MIMIC DATA LOCATION ###############
PATH_TO_RAW_MIMIC_DATA = "/mnt/d/mimic-4-1.0/physionet.org/files/mimiciv/1.0/"
####################################################################################################
####################################################################################################


# Module imports
import os

# Local imports
from mimic_pipeline.duck import configure_duckdb
from mimic_pipeline.io_registry import prepare_base_data
from mimic_pipeline.cohort import cohort_and_cleaning
from mimic_pipeline.timeseries import timeseries_creation

# Constants
CLEAR_DB_DATA = True
DATA_DIR = "data/"
BASE_OUTPUT_PATH = "data/base_parquet/"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Clear previous ddb file if needed
    if CLEAR_DB_DATA:
        ddb_path = os.path.join(BASE_OUTPUT_PATH, "data.db")
        if os.path.exists(ddb_path):
            os.remove(ddb_path)
        ddb_path = os.path.join(BASE_OUTPUT_PATH, "data.duckdb.wal")
        if os.path.exists(ddb_path):
            os.remove(ddb_path)

    # Prep ddb connection for data exploration
    ddb = configure_duckdb(database_name=DATA_DIR + "data.duckdb", memory_limit="8GB", temp_directory=BASE_OUTPUT_PATH)
    
    prepare_base_data(
        ddb,
        data_dir=DATA_DIR,
        base_csv_path=PATH_TO_RAW_MIMIC_DATA,
        base_output_path=BASE_OUTPUT_PATH
    )

    cohort_and_cleaning(
        ddb,
        base_output_path=BASE_OUTPUT_PATH
    )
    
    timeseries_creation(ddb)


if __name__ == "__main__":
    main()
