import duckdb

from .duck import save_parquet, peek
from .feature_selection import (
    build_zero_fill_select_body,
    combine_features,
    create_patient_bins,
    create_static_features,
    create_windowed_labels,
    downselect_chartevents,
    downselect_inputevents,
    downselect_outputevents,
    update_final_missing_values,
    zero_fill,
    chartevents_avg_features,
    inputevents_sum_features,
    outputevents_sum_features
)


# This is already done as a part of downselection process of the event tables
# The function that does this is create patient bins, and it is called timeseries_creation
def create_timeseries_bins(ddb: duckdb.DuckDBPyConnection, from_table = "ICUSTAYS"):
    query = f"""
    CREATE OR REPLACE TABLE ALL_PATIENT_BINS AS
    SELECT
        stay_id,
        intime,
        outtime,
        -- Use generate_series to create an index for each 4-hour bin
        -- Use UNNEST to expand the series into multiple rows
        -- The resulting 'time_bin_index' is the time step (0, 1, 2, ...)
        -- Subtract 1 is needed because the series is inclusive of the upper bound
        UNNEST(generate_series(
            0,
            (CEIL(DATE_DIFF('hour', intime, outtime) / 4.0))::INTEGER - 1
        )) AS time_bin_index,
        -- Calculate the start time for each bin (relative to intime)
        intime + INTERVAL '4' HOUR * time_bin_index AS bin_start_time,
        -- Calculate the end time for each bin
        intime + INTERVAL '4' HOUR * (time_bin_index + 1) AS bin_end_time
    FROM {from_table};
    """
    ddb.execute(query)

    save_parquet(ddb, out_dir="data/preprocessing_checkpoints/", table="ALL_PATIENT_BINS", filename="all_patient_bins.parquet")


def timeseries_creation(ddb: duckdb.DuckDBPyConnection):

    create_patient_bins(ddb, from_table="ICUSTAYS_COHORT_FIRST", output_table="ALL_PATIENT_BINS")
    
    downselect_chartevents(ddb, from_table="CLEAN_CHARTEVENTS", output_table="CHARTEVENTS_PIVOTED")
    zero_fill(ddb,
            select_body=build_zero_fill_select_body(chartevents_avg_features.values()),
            from_table="CHARTEVENTS_PIVOTED",
            output_table="CHARTEVENTS_ZEROED",
            print_query=False)
    
    downselect_outputevents(ddb, from_table="CLEAN_OUTPUTEVENTS", output_table="OUTPUTEVENTS_PIVOTED")
    zero_fill(ddb,
            select_body=build_zero_fill_select_body(outputevents_sum_features.values()),
            from_table="OUTPUTEVENTS_PIVOTED",
            output_table="OUTPUTEVENTS_ZEROED",
            print_query=False)

    downselect_inputevents(ddb, from_table="CLEAN_INPUTEVENTS", output_table="INPUTEVENTS_PIVOTED")
    zero_fill(ddb,
            select_body=build_zero_fill_select_body(inputevents_sum_features.values()),
            from_table="INPUTEVENTS_PIVOTED",
            output_table="INPUTEVENTS_ZEROED",
            print_query=False)
    
    create_static_features(ddb,
                        icu="ICUSTAYS_COHORT_FIRST",
                        admissions="ADMISSIONS",
                        patients="PATIENTS",
                        output_table="STATIC_FEATURES")
    
    combine_features(ddb,
                    bins="ALL_PATIENT_BINS",
                    static_features="STATIC_FEATURES",
                    ce="CHARTEVENTS_ZEROED",
                    oe="OUTPUTEVENTS_ZEROED",
                    ie="INPUTEVENTS_ZEROED",
                    output_table="FINAL_MEASUREMENT_TABLE",
                    print_query=False)
    
    update_final_missing_values(ddb, print_query=False)

    create_windowed_labels(ddb, icu="ICUSTAYS_COHORT_FIRST", cohort="COHORT", output_table="LABELS_33DAY")

    # Creating the final time series
    query = f"""
    CREATE OR REPLACE TABLE FINAL_TIMESERIES AS
    SELECT
        m.*,
        l.event_time_bin,
        l.event_flag
    FROM FINAL_MEASUREMENT_TABLE AS m
    LEFT JOIN LABELS_33DAY as l
        ON m.stay_id = l.stay_id;
    """
    ddb.execute(query)
    save_parquet(ddb, out_dir="../models/", table="FINAL_TIMESERIES", filename="final_timeseries.parquet")
    print("Final generated timeseries sample:")
    peek(ddb, "FINAL_TIMESERIES")


    tables = ddb.execute("PRAGMA show_tables;").fetchall()
    for t in tables:
        print(t[0])

