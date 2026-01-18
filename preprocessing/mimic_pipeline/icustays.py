import duckdb


def filter_icustays_on_cohort(ddb: duckdb.DuckDBPyConnection, icu="ICUSTAYS", cohort="COHORT", output_table="ICUSTAYS_COHORT"):
    query = f"""
        CREATE OR REPLACE TABLE {output_table} AS
        SELECT i.*
        FROM {icu} i
        JOIN {cohort} c
            USING (subject_id, hadm_id)
    """
    ddb.execute(query)
    print(f"Created {output_table}.")


def filter_icustays_first_stay(ddb: duckdb.DuckDBPyConnection, from_table="ICUSTAYS_COHORT_REDUCED_24H", output_table="ICUSTAYS_COHORT_FIRST"):
    query = f"""
        CREATE OR REPLACE TABLE {output_table} AS
        SELECT *
        FROM (
            SELECT i.*,
                ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime) AS rn
            FROM {from_table} i
        )
        WHERE rn = 1;
        """
    ddb.execute(query)


# Restrict to ICU stays with data in all three event tables within first 24 hours
def filter_icustays_first_24h(ddb: duckdb.DuckDBPyConnection, base_icustays_table: str, base_chartevents_table: str, base_outputevents_table: str, base_inputevents_table: str) -> None:
    # Based loosely on https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts_duckdb/firstday files

    # 24-hour restriction on CLEAN_CHARTEVENTS, CLEAN_OUTPUTEVENTS, CLEAN_INPUTEVENTS
    # There must be input, output, and chart events within the first 24 hours of ICU stay
    query = f"""
        CREATE OR REPLACE TABLE {base_icustays_table}_HAS_CE_24H AS
        SELECT i.stay_id
        FROM {base_icustays_table} i
        WHERE EXISTS (
        SELECT 1
        FROM {base_chartevents_table} ce
        WHERE ce.stay_id = i.stay_id
            AND ce.charttime >= i.intime
            AND ce.charttime <  i.intime + INTERVAL '24' HOUR
        LIMIT 1
        );
        """
    ddb.execute(query)

    query = f"""
        CREATE OR REPLACE TABLE {base_icustays_table}_HAS_OE_24H AS
        SELECT i.stay_id
        FROM {base_icustays_table} i
        WHERE EXISTS (
        SELECT 1
        FROM {base_outputevents_table} oe
        WHERE oe.stay_id = i.stay_id
            AND oe.charttime >= i.intime
            AND oe.charttime <  i.intime + INTERVAL '24' HOUR
        LIMIT 1
        );
        """
    ddb.execute(query)

    query = f"""
        CREATE OR REPLACE TABLE {base_icustays_table}_HAS_IE_24H AS
        SELECT i.stay_id
        FROM {base_icustays_table} i
        WHERE EXISTS (
        SELECT 1
        FROM {base_inputevents_table} ie
        WHERE ie.stay_id = i.stay_id
            AND ie.starttime >= i.intime
            AND ie.starttime <  i.intime + INTERVAL '24' HOUR
            AND COALESCE(ie.endtime, ie.starttime) > i.intime
        LIMIT 1
        );
        """
    ddb.execute(query)

    # Create reduced ICU stays table with 24-hour restriction
    query = f"""
        CREATE OR REPLACE TABLE {base_icustays_table}_REDUCED_24H AS
        SELECT i.*
        FROM {base_icustays_table} i
        JOIN {base_icustays_table}_HAS_CE_24H USING (stay_id)
        JOIN {base_icustays_table}_HAS_OE_24H USING (stay_id)
        JOIN {base_icustays_table}_HAS_IE_24H USING (stay_id);
        """
    ddb.execute(query)


# Restrict to ICU stays with data in all three event tables
def filter_icustays_with_all_events(ddb: duckdb.DuckDBPyConnection, base_icustays_table: str, base_chartevents_table: str, base_outputevents_table: str, base_inputevents_table: str) -> None:
    query = f"""
        CREATE OR REPLACE TABLE {base_icustays_table}_HAS_CE AS
        SELECT i.stay_id
        FROM {base_icustays_table} i
        WHERE EXISTS (
        SELECT 1
        FROM {base_chartevents_table} ce
        WHERE ce.stay_id = i.stay_id
            AND ce.charttime BETWEEN i.intime AND i.outtime
        LIMIT 1
        );
        """
    ddb.execute(query)

    query = f"""
        CREATE OR REPLACE TABLE {base_icustays_table}_HAS_OE AS
        SELECT i.stay_id
        FROM {base_icustays_table} i
        WHERE EXISTS (
        SELECT 1
        FROM {base_outputevents_table} oe
        WHERE oe.stay_id = i.stay_id
            AND oe.charttime BETWEEN i.intime AND i.outtime
        LIMIT 1
        );
        """
    ddb.execute(query)

    query = f"""
        CREATE OR REPLACE TABLE {base_icustays_table}_HAS_IE AS
        SELECT i.stay_id
        FROM {base_icustays_table} i
        WHERE EXISTS (
        SELECT 1
        FROM {base_inputevents_table} ie
        WHERE ie.stay_id = i.stay_id
            AND ie.starttime < i.outtime
            AND COALESCE(ie.endtime, ie.starttime) > i.intime
        LIMIT 1
        );
        """
    ddb.execute(query)

    query = f"""
        CREATE OR REPLACE TABLE {base_icustays_table}_REDUCED AS
        SELECT i.*
        FROM {base_icustays_table} i
        JOIN {base_icustays_table}_HAS_CE USING (stay_id)
        JOIN {base_icustays_table}_HAS_OE USING (stay_id)
        JOIN {base_icustays_table}_HAS_IE USING (stay_id);
        """
    ddb.execute(query)

    print(f"Filtered {base_icustays_table} to {base_icustays_table}_REDUCED with data in all three event tables.")


