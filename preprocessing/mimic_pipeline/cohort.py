import duckdb, os
from .duck import run, count, save_parquet
from .io_registry import register_parquet
from .icustays import filter_icustays_on_cohort, filter_icustays_first_24h, filter_icustays_first_stay


# Creates CLEAN_CHARTEVENTS
def clean_chartevents(ddb: duckdb.DuckDBPyConnection) -> None:
    ddb.execute("CREATE OR REPLACE TABLE CLEAN_CHARTEVENTS AS SELECT * FROM CHARTEVENTS;")

    # The authors of Survival Seq2seq keep only entries without warnings
    query = f"""
        DELETE FROM CLEAN_CHARTEVENTS
        WHERE warning IS NOT NULL AND warning <> 0
        """
    ddb.execute(query)
    ddb.execute("ALTER TABLE CLEAN_CHARTEVENTS DROP COLUMN warning;")

    # storetime column is not needed
    ddb.execute("ALTER TABLE CLEAN_CHARTEVENTS DROP COLUMN storetime;")

    save_parquet(ddb, out_dir="data/preprocessing_checkpoints/", table="CLEAN_CHARTEVENTS", filename="clean_chartevents.parquet")

# Clean INPUTEVENTS table
def clean_inputevents(ddb: duckdb.DuckDBPyConnection) -> None:
    ddb.execute("CREATE OR REPLACE TABLE CLEAN_INPUTEVENTS AS SELECT * FROM INPUTEVENTS;")
    # Not aware of any cleaning steps for this table yet
    save_parquet(ddb, out_dir="data/preprocessing_checkpoints/", table="CLEAN_INPUTEVENTS", filename="clean_inputevents.parquet")

# Clean OUTPUTEVENTS table
def clean_outputevents(ddb: duckdb.DuckDBPyConnection) -> None:
    ddb.execute("CREATE OR REPLACE TABLE CLEAN_OUTPUTEVENTS AS SELECT * FROM OUTPUTEVENTS;")
    # Not aware of any cleaning steps for this table yet
    save_parquet(ddb, out_dir="data/preprocessing_checkpoints/", table="CLEAN_OUTPUTEVENTS", filename="clean_outputevents.parquet")


# Cleaning the patient cohort takes a long time. This will save intermediate steps and log progress.
# It is also further optimized from the initial approach.
def build_cohort_with_checkpoints(ddb: duckdb.DuckDBPyConnection, out_dir: str, threads=14, mem_limit_gb=40) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Increase resource allocation for this one
    run(ddb, f"PRAGMA threads={threads};", "PRAGMA threads set")
    run(ddb, f"PRAGMA memory_limit='{mem_limit_gb}GB';", "PRAGMA memory limit set")
    temp_directory = os.path.expanduser('~/.duckdb_tmp')
    os.makedirs(temp_directory, exist_ok=True)
    run(ddb, f"PRAGMA temp_directory='{temp_directory}';", "PRAGMA temp directory set")
    run(ddb, "PRAGMA enable_progress_bar=true;", "Progress bar enabled")

    # 1. Create BASE table with admissions data and survival time in hours
    base_query = f"""
        CREATE OR REPLACE TABLE BASE AS
        SELECT
            a.subject_id,
            a.hadm_id,
            a.admittime,
            a.dischtime,
            DATE_DIFF('hour', a.admittime, COALESCE(a.deathtime, a.dischtime))::DOUBLE AS survival_time_hours,
            CASE WHEN a.deathtime IS NOT NULL THEN 1 ELSE 0 END AS died_within_icu_stay
        FROM ADMISSIONS a
        WHERE a.admittime IS NOT NULL
            AND a.dischtime IS NOT NULL
            AND DATE_DIFF('hour', a.admittime, COALESCE(a.deathtime, a.dischtime))::DOUBLE > 0;
    """
    run(ddb, base_query, "Step 1: Create BASE table.")
    count(ddb, "BASE", "Count rows in BASE")
    save_parquet(ddb, out_dir, "BASE", "base.parquet")

    # 2. Normalize dx titles once
    dd_norm_query = f"""
        CREATE OR REPLACE TABLE DD_NORM AS
        SELECT
            dd.icd_code,
            dd.icd_version,
            LOWER(dd.long_title) AS long_title
        FROM d_icd_diagnoses dd;
    """
    run(ddb, dd_norm_query, "Step 1: Create DD_NORM (Normalize titles as lowercase)")
    count(ddb, "DD_NORM", "Count rows in DD_NORM")
    save_parquet(ddb, out_dir, "DD_NORM", "dd_norm.parquet")

    # 3. Create EXCLUDE_PATTERNS table
    run(ddb, "CREATE OR REPLACE TEMP TABLE EXCLUDE_PATTERNS(pattern TEXT);")
    insert_exclude_patterns = f"""
        INSERT INTO EXCLUDE_PATTERNS VALUES
            ('%sids%'),
            ('%sudden infant death%'),
            ('%unattended death%'),
            ('%maternal death affecting fetus%'),
            ('%fetal death from asphyxia or anoxia during labor%'),
            ('%intrauterine death%');
    """
    run(ddb, insert_exclude_patterns, "Step 3: Insert exclude patterns in EXCLUDE_PATTERNS.")

    # 4. Create EXCLUDE_TITLES table
    exclude_titles_query = f"""
        CREATE OR REPLACE TABLE EXCLUDE_TITLES AS
        SELECT DISTINCT dd.icd_code, dd.icd_version
        FROM DD_NORM dd
        JOIN EXCLUDE_PATTERNS ep
            ON dd.long_title LIKE ep.pattern;
    """
    run(ddb, exclude_titles_query, "Step 4: Create EXCLUDE_TITLES table.")
    count(ddb, "EXCLUDE_TITLES", "Count rows in EXCLUDE_TITLES")
    save_parquet(ddb, out_dir, "EXCLUDE_TITLES", "exclude_titles.parquet")

    # 5. Create BAD_ADMISSIONS table with admission IDs having dxs to exclude
    bad_admissions_query = f"""
        CREATE OR REPLACE TABLE BAD_ADMISSIONS AS
        SELECT DISTINCT d.hadm_id
        FROM diagnoses_icd d
        JOIN EXCLUDE_TITLES et
            ON d.icd_code = et.icd_code
            AND d.icd_version = et.icd_version;
    """
    run(ddb, bad_admissions_query, "Step 5: Create BAD_ADMISSIONS table with admission IDs having dxs to exclude.")
    count(ddb, "BAD_ADMISSIONS", "Count rows in BAD_ADMISSIONS")
    save_parquet(ddb, out_dir, "BAD_ADMISSIONS", "bad_admissions.parquet")

    # 6. Create final COHORT table excluding bad admissions
    # ANTI JOIN does not work
    cohort_query = f"""
        CREATE OR REPLACE TABLE COHORT AS
        SELECT b.*
        FROM BASE b
        WHERE NOT EXISTS (
            SELECT 1
            FROM BAD_ADMISSIONS ba
            WHERE b.hadm_id = ba.hadm_id
        );
    """
    # Equivalent to:
    # LEFT ANTI JOIN BAD_ADMISSIONS ba
    #         ON b.hadm_id = ba.hadm_id;

    run(ddb, cohort_query, "Step 6: Create final COHORT table excluding bad admissions.")
    count(ddb, "COHORT", "Count rows in COHORT")
    save_parquet(ddb, out_dir, "COHORT", "cohort.parquet")

    print("Cohort building with checkpoints complete.")

    run(ddb, "RESET threads;", "Reset to automatic parallelism")
    run(ddb, "PRAGMA memory_limit='8GB';", "Reset to original memory limit")
    run(ddb, "RESET temp_directory;", "Reset temp directory to default")
    run(ddb, "PRAGMA enable_progress_bar=false;", "Progress bar disabled")


def validate_cohort(ddb: duckdb.DuckDBPyConnection) -> None:
        print("Counts:")
        ddb.query("""
            SELECT
                (SELECT COUNT(*) FROM BASE) AS base_cnt,
                (SELECT COUNT(*) FROM BAD_ADMISSIONS) AS bad_cnt,
                (SELECT COUNT(*) FROM COHORT) AS cohort_cnt,
                (SELECT COUNT(*) FROM BASE) - (SELECT COUNT(*) FROM BAD_ADMISSIONS) AS expected
        """).show()

        print("Overlaps between COHORT and BAD_ADMISSIONS (must be 0):")
        ddb.query("""
            SELECT COUNT(*) AS overlaps
            FROM COHORT c
            JOIN BAD_ADMISSIONS b ON c.hadm_id = b.hadm_id
        """).show()

        print("Sample diagnoses from BAD_ADMISSIONS:")
        ddb.query("""
            SELECT DISTINCT
                d.hadm_id,
                dn.long_title
            FROM BAD_ADMISSIONS ba
            JOIN diagnoses_icd d
                ON ba.hadm_id = d.hadm_id
            JOIN DD_NORM dn
                ON dn.icd_code = d.icd_code
                AND dn.icd_version = d.icd_version
            ORDER BY d.hadm_id
        """).show()


def cohort_and_cleaning(ddb: duckdb.DuckDBPyConnection, base_output_path = "data/base_parquet/"):
    # Data to process
    directory_map = {
        # core
        "ADMISSIONS": "admissions.parquet",
        "PATIENTS": "patients.parquet",

        # hospital
        "DIAGNOSES_ICD": "diagnoses_icd.parquet",
        "D_ICD_DIAGNOSES": "d_icd_diagnoses.parquet",

        # icu
        "ICUSTAYS": "icustays.parquet",
        "INPUTEVENTS": "inputevents.parquet",
        "OUTPUTEVENTS": "outputevents.parquet",
        "CHARTEVENTS": "chartevents.parquet",
        "D_ITEMS": "d_items.parquet",
    }
    filepath_map = {
        key: base_output_path + value for key, value in directory_map.items()
    }

    print("Creating relevant start tables...")
    register_parquet(ddb, filepath_map)

    # Build out the patient cohort from the initial files
    build_cohort_with_checkpoints(ddb, out_dir="data/preprocessing_checkpoints/", threads=14, mem_limit_gb=40)
    print("Generated cohort:")
    ddb.query("SELECT COUNT(*) AS cohort_size FROM COHORT").show()

    print("Cleaning tables...")
    clean_chartevents(ddb)
    clean_inputevents(ddb)
    clean_outputevents(ddb)

    # Limit the ICUSTAYS data by the patient COHORT
    filter_icustays_on_cohort(ddb, icu="ICUSTAYS", cohort="COHORT", output_table="ICUSTAYS_COHORT")

    # Filter ICUSTAYS_COHORT to only those with data in all three event tables
    # AND restrict to ICU stays with data in all three event tables within first 24 hours
    filter_icustays_first_24h(ddb, "ICUSTAYS_COHORT", "CLEAN_CHARTEVENTS", "CLEAN_OUTPUTEVENTS", "CLEAN_INPUTEVENTS")

    filter_icustays_first_stay(ddb, from_table="ICUSTAYS_COHORT_REDUCED_24H", output_table="ICUSTAYS_COHORT_FIRST")
    # The dataset is already restricted to adults

