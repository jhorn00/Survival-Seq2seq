# Preprocessing script for MIMIC-IV 1.0 data

####################################################################################################
############################################ Constants #############################################
PATH_TO_RAW_MIMIC_DATA = "/mnt/d/mimic-4-1.0/physionet.org/files/mimiciv/1.0/" # Download location
BASE_OUTPUT_PATH = "data/base_parquet/" # Where do you want intermediate tables/files?
CLEAR_DB_DATA = True # Delete existing duckdb files in BASE_OUTPUT_PATH
DATA_DIR = "data/" # Where you want intermediate and preprocessed data to be written
# TODO: Use these constants and pass through to called functions
DDB_DATABASE_NAME = DATA_DIR + "data.duckdb"
DDB_MEMORY_LIMIT = "8GB" # GB or % of memory usage before duckdb spills to disk temp files
DDB_TEMP_DIRECTORY = BASE_OUTPUT_PATH # Location of duckdb temp dir for "swap" spill files
####################################################################################################
####################################################################################################


# Module imports
import os

# Local imports
from mimic_pipeline.duck import configure_duckdb
from mimic_pipeline.io_registry import prepare_base_data
from mimic_pipeline.cohort import cohort_and_cleaning
from mimic_pipeline.timeseries import timeseries_creation


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1
    # Clear previous ddb file if needed - we have not directly identified any issues if you fail to do this and overwrites occur
    if CLEAR_DB_DATA:
        ddb_path = os.path.join(BASE_OUTPUT_PATH, "data.db")
        if os.path.exists(ddb_path):
            os.remove(ddb_path)
        ddb_path = os.path.join(BASE_OUTPUT_PATH, "data.duckdb.wal")
        if os.path.exists(ddb_path):
            os.remove(ddb_path)

    # 2
    # Create and configure db connection
    ddb = configure_duckdb(database_name=DATA_DIR + "data.duckdb", memory_limit="8GB", temp_directory=BASE_OUTPUT_PATH)
    
    # 3
    # 
    # TODO: Remove duplicate connection made in this function and verify results
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
