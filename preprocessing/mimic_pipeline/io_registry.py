import duckdb
import os
from typing import Dict

# Local imports
from .duck import run, configure_duckdb

# (Parquet) Register tables in db once
def register_parquet(ddb: duckdb.DuckDBPyConnection, filepath_map: Dict[str, str]) -> None:
    # Because of the data format we need to specify quote and escape characters
    run(ddb, f"CREATE OR REPLACE TABLE ADMISSIONS AS SELECT * FROM '{filepath_map['ADMISSIONS']}'", "Loaded ADMISSIONS table")
    run(ddb, f"CREATE OR REPLACE TABLE PATIENTS AS SELECT * FROM '{filepath_map['PATIENTS']}'", "Loaded PATIENTS table")

    run(ddb, f"CREATE OR REPLACE TABLE DIAGNOSES_ICD AS SELECT * FROM '{filepath_map['DIAGNOSES_ICD']}'", "Loaded DIAGNOSES_ICD table")
    run(ddb, f"CREATE OR REPLACE TABLE D_ICD_DIAGNOSES AS SELECT * FROM '{filepath_map['D_ICD_DIAGNOSES']}'", "Loaded D_ICD_DIAGNOSES table")

    run(ddb, f"CREATE OR REPLACE TABLE ICUSTAYS AS SELECT * FROM '{filepath_map['ICUSTAYS']}'", "Loaded ICUSTAYS table")
    run(ddb, f"CREATE OR REPLACE TABLE INPUTEVENTS AS SELECT * FROM '{filepath_map['INPUTEVENTS']}'", "Loaded INPUTEVENTS table")
    run(ddb, f"CREATE OR REPLACE TABLE OUTPUTEVENTS AS SELECT * FROM '{filepath_map['OUTPUTEVENTS']}'", "Loaded OUTPUTEVENTS table")
    run(ddb, f"CREATE OR REPLACE TABLE CHARTEVENTS AS SELECT * FROM '{filepath_map['CHARTEVENTS']}'", "Loaded CHARTEVENTS table")
    run(ddb, f"CREATE OR REPLACE TABLE D_ITEMS AS SELECT * FROM '{filepath_map['D_ITEMS']}'", "Loaded D_ITEMS table")

def register_parquet_general(ddb: duckdb.DuckDBPyConnection, filepath_map: Dict[str, str]) -> None:
    for table_name, filepath in filepath_map.items():
        run(ddb, f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM '{filepath}'", f"Loaded {table_name} table")

# (csv.gz) Register tables in db once
def register_csv_gz(ddb: duckdb.DuckDBPyConnection, filepath_map: Dict[str, str]) -> None:
    # Because of the data format we need to specify quote and escape characters
    run(ddb, f"CREATE OR REPLACE TABLE ADMISSIONS AS SELECT * FROM read_csv_auto('{filepath_map['ADMISSIONS']}', types={{'admittime':'TIMESTAMP','dischtime':'TIMESTAMP','deathtime':'TIMESTAMP'}}, quote='\"', escape='\"')", "Created ADMISSIONS table")
    run(ddb, f"CREATE OR REPLACE TABLE PATIENTS AS SELECT * FROM read_csv_auto('{filepath_map['PATIENTS']}', quote='\"', escape='\"')", "Created PATIENTS table")

    run(ddb, f"CREATE OR REPLACE TABLE DIAGNOSES_ICD AS SELECT * FROM read_csv_auto('{filepath_map['DIAGNOSES_ICD']}', quote='\"', escape='\"')", "Created DIAGNOSES_ICD table")
    run(ddb, f"CREATE OR REPLACE TABLE D_ICD_DIAGNOSES AS SELECT * FROM read_csv_auto('{filepath_map['D_ICD_DIAGNOSES']}', quote='\"', escape='\"')", "Created D_ICD_DIAGNOSES table")

    run(ddb, f"CREATE OR REPLACE TABLE ICUSTAYS AS SELECT * FROM read_csv_auto('{filepath_map['ICUSTAYS']}', types={{'intime':'TIMESTAMP','outtime':'TIMESTAMP'}}, quote='\"', escape='\"')", "Created ICUSTAYS table")
    run(ddb, f"CREATE OR REPLACE TABLE INPUTEVENTS AS SELECT * FROM read_csv_auto('{filepath_map['INPUTEVENTS']}', types={{'starttime':'TIMESTAMP','endtime':'TIMESTAMP'}}, quote='\"', escape='\"')", "Created INPUTEVENTS table")
    run(ddb, f"CREATE OR REPLACE TABLE OUTPUTEVENTS AS SELECT * FROM read_csv_auto('{filepath_map['OUTPUTEVENTS']}', types={{'charttime':'TIMESTAMP'}}, quote='\"', escape='\"')", "Created OUTPUTEVENTS table")
    run(ddb, f"CREATE OR REPLACE TABLE CHARTEVENTS AS SELECT * FROM read_csv_auto('{filepath_map['CHARTEVENTS']}', types={{'charttime':'TIMESTAMP','storetime':'TIMESTAMP','value':'VARCHAR'}}, quote='\"', escape='\"')", "Created CHARTEVENTS table")  # Force value to string to avoid type issues
    run(ddb, f"CREATE OR REPLACE TABLE D_ITEMS AS SELECT * FROM read_csv_auto('{filepath_map['D_ITEMS']}', quote='\"', escape='\"')", "Created D_ITEMS table")

# Make parquet files from raw csvs
def csv_to_parquet(ddb: duckdb.DuckDBPyConnection, filepath_map: dict, out_dir = "data/base_parquet/") -> None:
    os.makedirs(out_dir, exist_ok=True)
    
    def cp(sql_select, out_name):
        copy_sql = f"""
        COPY ({sql_select})
        TO '{os.path.join(out_dir, out_name)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        run(ddb, copy_sql, f"Created {out_name}")

    # Because of the data format we need to specify quote and escape characters
    sql = f"SELECT * FROM read_csv_auto('{filepath_map['ADMISSIONS']}', types={{'admittime':'TIMESTAMP','dischtime':'TIMESTAMP','deathtime':'TIMESTAMP'}}, quote='\"', escape='\"')"
    cp(sql_select=sql, out_name="admissions.parquet")
    sql = f"SELECT * FROM read_csv_auto('{filepath_map['PATIENTS']}', quote='\"', escape='\"')"
    cp(sql_select=sql, out_name="patients.parquet")

    sql = f"SELECT * FROM read_csv_auto('{filepath_map['DIAGNOSES_ICD']}', quote='\"', escape='\"')"
    cp(sql_select=sql, out_name="diagnoses_icd.parquet")
    sql = f"SELECT * FROM read_csv_auto('{filepath_map['D_ICD_DIAGNOSES']}', quote='\"', escape='\"')"
    cp(sql_select=sql, out_name="d_icd_diagnoses.parquet")

    sql = f"SELECT * FROM read_csv_auto('{filepath_map['ICUSTAYS']}', types={{'intime':'TIMESTAMP','outtime':'TIMESTAMP'}}, quote='\"', escape='\"')"
    cp(sql_select=sql, out_name="icustays.parquet")
    sql = f"SELECT * FROM read_csv_auto('{filepath_map['INPUTEVENTS']}', types={{'starttime':'TIMESTAMP','endtime':'TIMESTAMP'}}, quote='\"', escape='\"')"
    cp(sql_select=sql, out_name="inputevents.parquet")
    sql = f"SELECT * FROM read_csv_auto('{filepath_map['OUTPUTEVENTS']}', types={{'charttime':'TIMESTAMP'}}, quote='\"', escape='\"')"
    cp(sql_select=sql, out_name="outputevents.parquet")
    sql = f"SELECT * FROM read_csv_auto('{filepath_map['CHARTEVENTS']}', types={{'charttime':'TIMESTAMP','storetime':'TIMESTAMP','value':'VARCHAR'}}, quote='\"', escape='\"')"
    cp(sql_select=sql, out_name="chartevents.parquet")
    sql = f"SELECT * FROM read_csv_auto('{filepath_map['D_ITEMS']}', quote='\"', escape='\"')"
    cp(sql_select=sql, out_name="d_items.parquet")


def prepare_base_data(ddb: duckdb.DuckDBPyConnection, data_dir = "data/", base_csv_path = "mimic_iv/", base_output_path = "data/base_parquet/"):
    os.makedirs(data_dir, exist_ok=True)

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
        key: base_csv_path + value for key, value in directory_map.items()
    }

    ddb = configure_duckdb(database_name="data/data.duckdb", memory_limit="8GB", temp_directory=base_csv_path)
    csv_to_parquet(ddb, filepath_map, out_dir=base_output_path)

