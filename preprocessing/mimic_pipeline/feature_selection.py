import duckdb

from .duck import save_parquet

###### Target Features ######

# (itemid: 'label')
# Features from CHARTEVENTS, aggregated by AVG

# (itemid: 'label')

# Features from CHARTEVENTS, aggregated by AVG
# old
# chartevents_avg_features = {
#     220045: 'HeartRate',
#     220210: 'RespRate',
#     220277: 'O2Sat',
#     220179: 'SysBP_NonInvasive',
#     220180: 'DiasBP_NonInvasive',
#     220181: 'MeanBP_NonInvasive',
# }

# new
chartevents_avg_features = {
    # core vital
    220045: 'HeartRate',
    220210: 'RespRate',
    220277: 'SpO2',
    223762: 'Temp_C',
    220181: 'NIBP_Mean',
    220179: 'NIBP_Systolic',
    220180: 'NIBP_Diastolic',
    220052: 'ABP_Mean',
    220050: 'ABP_Systolic',
    220051: 'ABP_Diastolic',
    220074: 'CVP',
    228640: 'EtCO2',
    226329: 'BloodTemp_CCO',

    # respiratory and support
    223835: 'FiO2',
    223834: 'O2Flow',
    224685: 'TidalVolume_Observed',
    224686: 'TidalVolume_Spont',
    224684: 'TidalVolume_Set',
    224689: 'RespRate_Spont',
    224690: 'RespRate_Total',
    224687: 'MinuteVolume',
    224700: 'PEEP_Total',
    220339: 'PEEP_Set',
    224697: 'MeanAirwayPressure',
    224695: 'PeakInspiratoryPressure',
    224696: 'PlateauPressure',
    224738: 'InspiratoryTime',
    226871: 'ExpiratoryRatio',

    # labs, gas exchange
    225664: 'Glucose_Fingerstick',
    220621: 'Glucose_Serum',
    226537: 'Glucose_WholeBlood',
    220635: 'Magnesium',
    227073: 'AnionGap',
    220645: 'Sodium_Serum',
    220602: 'Chloride_Serum',
    225698: 'TCO2_Arterial',
    220615: 'Creatinine',
    225624: 'BUN',
    225677: 'Phosphate',
    220546: 'WBC',
    220228: 'Hemoglobin',
    226540: 'Hematocrit_WholeBlood',
    225667: 'Calcium_Ionized',
    220644: 'ALT',
    223830: 'pH_Arterial',
    220235: 'PaCO2_Arterial',
    220224: 'PaO2_Arterial',
    220227: 'SaO2_Arterial',
    220274: 'pH_Venous',
    225668: 'Lactate',

    # anthropometrics
    226512: 'AdmissionWeight_kg',
    226730: 'Height_cm',

    # hemodynamics
    223772: 'SvO2',
    224842: 'CardiacOutput_CCO',
    227543: 'CardiacOutput_Art',
    227546: 'SVV_Arterial',
    220060: 'PAP_Diastolic',
    220059: 'PAP_Systolic',
    220061: 'PAP_Mean',
    220765: 'ICP',
    227066: 'CPP',
    220088: 'CardiacOutput_Thermo',

    # dialysis, CRRT parameters
    226457: 'UltrafiltrateOutput',
    224191: 'HourlyPatientFluidRemoval',
    224144: 'Dialysis_BloodFlow',
    224150: 'Dialysis_FilterPressure',
    224149: 'Dialysis_AccessPressure',
    224152: 'Dialysis_ReturnPressure',
    224151: 'Dialysis_EffluentPressure',
    224154: 'DialysateRate',
    224153: 'ReplacementRate',
    228004: 'CitrateRate',
    228005: 'PBP_ReplacementRate',
    228006: 'PostFilter_ReplacementRate',
    229247: 'TransmembranePressure',
    229248: 'PressureDrop',
}

# Features from OUTPUTEVENTS, aggregated by SUM
# old
# outputevents_sum_features = {
#     226559: 'UrineOutput_Foley',
# }
outputevents_sum_features = {
    # urine
    226559: 'UrineOutput_Foley',
    226560: 'UrineOutput_Void',
    226561: 'UrineOutput_CondomCath',
    226567: 'UrineOutput_StraightCath',
    226563: 'UrineOutput_Suprapubic',
    226627: 'UrineOutput_OR',

    # GI, stool, feeding
    226575: 'NG_Output',
    226576: 'OG_Output',
    226582: 'Ostomy_Output',
    226579: 'Stool_Output',
    226580: 'FecalBag_Output',
    227510: 'TFResidual_Output',

    # major drains
    226588: 'ChestTube1_Output',
    226599: 'JP1_DrainOutput',
}

# Features from INPUTEVENTS, aggregated by SUM
## old
# inputevents_sum_features = {
#     225158: 'NaCl_0_9_Bolus', # 0.9% Normal Saline
#     221906: 'Norepinephrine',
# }

inputevents_sum_features = {
    # fluid, bolus, intake
    225158: 'NaCl_0_9_Bolus',          # 0.9% Normal Saline
    220949: 'Dextrose_5_Fluid',
    225943: 'Solution_Generic_Fluid',
    225828: 'LR_Fluid',
    226452: 'PO_Intake',
    226453: 'GT_Flush',
    226089: 'Piggyback_Fluid',

    # key meds
    222168: 'Propofol_Dose',
    221906: 'Norepinephrine_Dose',
    221749: 'Phenylephrine_Dose',
    221744: 'Fentanyl_Dose',
    221668: 'Midazolam_Dose',
    221833: 'Hydromorphone_Dose',
    221794: 'Furosemide_Dose',
    223258: 'InsulinRegular_Dose',
    225975: 'HeparinProphylaxis_Dose',
    225798: 'Vancomycin_Dose',
    225166: 'PotassiumChloride_Dose',
    227522: 'KCL_Bolus_mL',
}

#############################


###### Downselect and bin data ######
def create_patient_bins(ddb: duckdb.DuckDBPyConnection, from_table="ICUSTAYS", output_table="ALL_PATIENT_BINS"):
    query = f"""
    CREATE OR REPLACE TABLE {output_table} AS
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
    print(f"Created {output_table}.")
    save_parquet(ddb, out_dir="data/preprocessing_checkpoints", table=output_table, filename=f"{output_table.lower()}.parquet")


def downselect_chartevents(ddb: duckdb.DuckDBPyConnection, from_table="CLEAN_CHARTEVENTS", output_table="CHARTEVENTS_PIVOTED"):
    # Create the `itemid` list for the SQL IN clause
    ce_avg_ids = ', '.join([str(k) for k in chartevents_avg_features.keys()])
    # Create the PIVOT list (e.g., 220045 AS "HeartRate", ...)
    ce_pivot_list = ', '.join([f"{k} AS \"{v}\"" for k, v in chartevents_avg_features.items()])

    query = f"""
    CREATE OR REPLACE TABLE {output_table} AS
    SELECT * FROM (
        SELECT
            ce.stay_id,
            -- Calculate which 4-hour bin this event falls into
            FLOOR(DATE_DIFF('hour', i.intime, ce.charttime) / 4.0)::INTEGER AS time_bin_index,
            ce.itemid,
            ce.valuenum -- The numeric value to average
        FROM {from_table} ce
        JOIN ICUSTAYS i ON ce.stay_id = i.stay_id
        WHERE
            ce.itemid IN ({ce_avg_ids})
            AND ce.valuenum IS NOT NULL
    )
    PIVOT (
        AVG(valuenum) -- Use AVG for these features
        FOR itemid IN ({ce_pivot_list})
    );
    """
    ddb.execute(query)
    print(f"Created {output_table}.")
    save_parquet(ddb, out_dir="data/preprocessing_checkpoints", table=output_table, filename=f"{output_table.lower()}.parquet")


def downselect_outputevents(ddb: duckdb.DuckDBPyConnection, from_table="CLEAN_OUTPUTEVENTS", output_table="OUTPUTEVENTS_PIVOTED"):
    # Create the `itemid` list for the SQL IN clause
    oe_sum_ids = ', '.join([str(k) for k in outputevents_sum_features.keys()])
    # Create the PIVOT list (e.g., 220045 AS "HeartRate", ...)
    oe_pivot_list = ', '.join([f"{k} AS \"{v}\"" for k, v in outputevents_sum_features.items()])

    query = f"""
    CREATE OR REPLACE TABLE {output_table} AS
    SELECT * FROM (
        SELECT
            oe.stay_id,
            FLOOR(DATE_DIFF('hour', i.intime, oe.charttime) / 4.0)::INTEGER AS time_bin_index,
            oe.itemid,
            oe.value -- The numeric value you want to sum
        FROM {from_table} oe
        JOIN ICUSTAYS i ON oe.stay_id = i.stay_id
        WHERE
            oe.itemid IN ({oe_sum_ids})
            AND oe.value IS NOT NULL
    )
    PIVOT (
        SUM(value) -- Use SUM for these features
        FOR itemid IN ({oe_pivot_list})
    );
    """
    ddb.execute(query)
    print(f"Created {output_table}.")
    save_parquet(ddb, out_dir="data/preprocessing_checkpoints", table=output_table, filename=f"{output_table.lower()}.parquet")


def downselect_inputevents(ddb: duckdb.DuckDBPyConnection, from_table="CLEAN_INPUTEVENTS", output_table="INPUTEVENTS_PIVOTED"):
    # Create the `itemid` list for the SQL IN clause
    ie_sum_ids = ', '.join([str(k) for k in inputevents_sum_features.keys()])
    # Create the PIVOT list (e.g., 220045 AS "HeartRate", ...)
    ie_pivot_list = ', '.join([f"{k} AS \"{v}\"" for k, v in inputevents_sum_features.items()])

    query = f"""
    CREATE OR REPLACE TABLE {output_table} AS
    SELECT * FROM (
        SELECT
            ie.stay_id,
            FLOOR(DATE_DIFF('hour', i.intime, ie.starttime) / 4.0)::INTEGER AS time_bin_index,
            ie.itemid,
            ie.amount -- The numeric value you want to sum
        FROM {from_table} ie
        JOIN ICUSTAYS i ON ie.stay_id = i.stay_id
        WHERE
            ie.itemid IN ({ie_sum_ids})
            AND ie.amount IS NOT NULL
    )
    PIVOT (
        SUM(amount) -- Use SUM for these features
        FOR itemid IN ({ie_pivot_list})
    );
    """
    ddb.execute(query)
    print(f"Created {output_table}.")
    save_parquet(ddb, out_dir="data/preprocessing_checkpoints", table=output_table, filename=f"{output_table.lower()}.parquet")


#####################################
#####################################


def build_zero_fill_select_body(feature_names):
    parts = []
    for v in feature_names:
        parts.append(f"COALESCE({v}, 0.0) AS {v}_value")
        parts.append(f"CASE WHEN {v} IS NULL THEN 1 ELSE 0 END AS {v}_missing")
    return ",\n".join(parts)


def zero_fill(ddb: duckdb.DuckDBPyConnection, select_body: str, from_table: str, output_table: str, print_query=False):
    query = f"""
    CREATE OR REPLACE TABLE {output_table} AS
    SELECT
        stay_id,
        time_bin_index,
        {select_body}
    FROM {from_table};
    """
    if print_query:
        print(f"Query:\n{query}")
    ddb.execute(query)
    print(f"Created {output_table}.")
    save_parquet(ddb, out_dir="data/preprocessing_checkpoints", table=output_table, filename=f"{output_table.lower()}.parquet")


def create_static_features(ddb: duckdb.DuckDBPyConnection, icu="ICUSTAYS", admissions="ADMISSIONS", patients="PATIENTS", output_table="STATIC_FEATURES"):
    query = f"""
    CREATE OR REPLACE TABLE {output_table} AS
    SELECT
        i.stay_id,
        -- Calculate age at admission
        (EXTRACT(YEAR FROM a.admittime) - p.anchor_year + p.anchor_age) AS admission_age,
        -- One-hot encode gender
        CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS gender_M,
        CASE WHEN p.gender = 'F' THEN 1 ELSE 0 END AS gender_F
    FROM {icu} i
    JOIN {admissions} a ON i.hadm_id = a.hadm_id
    JOIN {patients} p ON i.subject_id = p.subject_id;
    """
    ddb.execute(query)
    print(f"Created {output_table}.")
    save_parquet(ddb, out_dir="data/preprocessing_checkpoints", table=output_table, filename=f"{output_table.lower()}.parquet")


def build_combine_select_body(prefix, feature_names):
    parts = []
    for v in feature_names:
        parts.append(f"{prefix}.{v}_value")
        parts.append(f"{prefix}.{v}_missing")
    return ",\n".join(parts)


def combine_features(ddb: duckdb.DuckDBPyConnection,
                     bins="ALL_PATIENT_BINS",
                     static_features="STATIC_FEATURES",
                     ce="CHARTEVENTS_ZEROED",
                     oe="OUTPUTEVENTS_ZEROED",
                     ie="INPUTEVENTS_ZEROED",
                     output_table="FINAL_MEASUREMENT_TABLE",
                     print_query=False):
    query = f"""
    CREATE OR REPLACE TABLE {output_table} AS
    SELECT
        b.stay_id,
        b.time_bin_index,
        
        -- Static features (will be duplicated for each bin)
        s.admission_age,
        s.gender_M,
        s.gender_F,
        
        -- Dynamic CHARTEVENTS features
        {build_combine_select_body("ce", chartevents_avg_features.values())},
        
        -- Dynamic OUTPUTEVENTS features
        {build_combine_select_body("oe", outputevents_sum_features.values())},
        
        -- Dynamic INPUTEVENTS features
        {build_combine_select_body("ie", inputevents_sum_features.values())}
        
    FROM {bins} b
    -- Join static features
    LEFT JOIN {static_features} s ON b.stay_id = s.stay_id
    -- Join pivoted CHARTEVENTS
    LEFT JOIN {ce} ce ON b.stay_id = ce.stay_id AND b.time_bin_index = ce.time_bin_index
    -- Join pivoted OUTPUTEVENTS
    LEFT JOIN {oe} oe ON b.stay_id = oe.stay_id AND b.time_bin_index = oe.time_bin_index
    -- Join pivoted INPUTEVENTS
    LEFT JOIN {ie} ie ON b.stay_id = ie.stay_id AND b.time_bin_index = ie.time_bin_index

    ORDER BY b.stay_id, b.time_bin_index;
    """
    if print_query:
        print(f"Query:\n{query}")
    ddb.execute(query)
    print(f"Created {output_table}.")
    save_parquet(ddb, out_dir="data/preprocessing_checkpoints", table=output_table, filename=f"{output_table.lower()}.parquet")


# def update_final_missing_values(ddb: duckdb.DuckDBPyConnection, print_query=False):
#     target_table = "FINAL_MEASUREMENT_TABLE"
#     query = f"""
#     UPDATE {target_table}
#     SET
#         HeartRate_value = COALESCE(HeartRate_value, 0),
#         HeartRate_missing = CASE WHEN HeartRate_value IS NULL THEN 1 ELSE HeartRate_missing END,

#         RespRate_value = COALESCE(RespRate_value, 0),
#         RespRate_missing = CASE WHEN RespRate_value IS NULL THEN 1 ELSE RespRate_missing END,

#         O2Sat_value = COALESCE(O2Sat_value, 0),
#         O2Sat_missing = CASE WHEN O2Sat_value IS NULL THEN 1 ELSE O2Sat_missing END,

#         SysBP_NonInvasive_value = COALESCE(SysBP_NonInvasive_value, 0),
#         SysBP_NonInvasive_missing = CASE WHEN SysBP_NonInvasive_value IS NULL THEN 1 ELSE SysBP_NonInvasive_missing END,

#         DiasBP_NonInvasive_value = COALESCE(DiasBP_NonInvasive_value, 0),
#         DiasBP_NonInvasive_missing = CASE WHEN DiasBP_NonInvasive_value IS NULL THEN 1 ELSE DiasBP_NonInvasive_missing END,

#         MeanBP_NonInvasive_value = COALESCE(MeanBP_NonInvasive_value, 0),
#         MeanBP_NonInvasive_missing = CASE WHEN MeanBP_NonInvasive_value IS NULL THEN 1 ELSE MeanBP_NonInvasive_missing END,

#         UrineOutput_Foley_value = COALESCE(UrineOutput_Foley_value, 0),
#         UrineOutput_Foley_missing = CASE WHEN UrineOutput_Foley_value IS NULL THEN 1 ELSE UrineOutput_Foley_missing END,

#         NaCl_0_9_Bolus_value = COALESCE(NaCl_0_9_Bolus_value, 0),
#         NaCl_0_9_Bolus_missing = CASE WHEN NaCl_0_9_Bolus_value IS NULL THEN 1 ELSE NaCl_0_9_Bolus_missing END,

#         Norepinephrine_value = COALESCE(Norepinephrine_value, 0),
#         Norepinephrine_missing = CASE WHEN Norepinephrine_value IS NULL THEN 1 ELSE Norepinephrine_missing END;
#     """
#     if print_query:
#         print(f"Query:\n{query}")
#     ddb.execute(query)
#     print(f"Created {target_table}.")
#     save_parquet(ddb, out_dir="data/preprocessing_checkpoints", table=target_table, filename=f"{target_table.lower()}.parquet")

def update_final_missing_values(ddb: duckdb.DuckDBPyConnection, print_query=False):
    target_table = "FINAL_MEASUREMENT_TABLE"
    all_feature_names = (list(chartevents_avg_features.values()) + list(inputevents_sum_features.values()) + list(outputevents_sum_features.values()))
    expressions = []
    for feature in all_feature_names:
        expressions.append(
            f"{feature}_missing = CASE WHEN {feature}_value IS NULL THEN 1 ELSE {feature}_missing END"
        )
        expressions.append(
            f"{feature}_value = COALESCE({feature}_value, 0)"
        )
        
    set_expression = ",\n".join(expressions)

    query = f"""
    UPDATE {target_table}
    SET
    {set_expression};
    """
    if print_query:
        print(f"Query:\n{query}")
    ddb.execute(query)
    print(f"Created {target_table}.")
    save_parquet(ddb, out_dir="data/preprocessing_checkpoints", table=target_table, filename=f"{target_table.lower()}.parquet")
    

def create_labels(ddb: duckdb.DuckDBPyConnection, icu="ICUSTAYS", cohort="COHORT", output_table="LABELS"):
    query = f"""
    CREATE OR REPLACE TABLE {output_table} AS
    SELECT
        ic.stay_id,
        c.died_within_icu_stay AS event_flag,
        c.survival_time_hours,
        -- Discretize the survival time into the same 4-hour bins
        -- This bin index is the 'tau_t' the model needs for loss
        FLOOR(c.survival_time_hours / 4.0)::INTEGER AS event_time_bin
    FROM
        {icu} ic
    JOIN
        {cohort} c ON ic.hadm_id = c.hadm_id;
    """
    ddb.execute(query)
    print(f"Created {output_table}.")
    save_parquet(ddb, out_dir="data/preprocessing_checkpoints", table=output_table, filename=f"{output_table.lower()}.parquet")


def create_windowed_labels(ddb: duckdb.DuckDBPyConnection, icu="ICUSTAYS", cohort="COHORT", output_table="LABELS_33DAY"):
    query = f"""
    -- 33 days * 24 hours/day = 792 hours
    -- 792 hours / 4 hours/bin = 198 bins (0-indexed to 197)
    CREATE OR REPLACE TABLE {output_table} AS
    SELECT
        ic.stay_id,
        c.survival_time_hours,
        FLOOR(c.survival_time_hours / 4.0)::INTEGER AS original_event_time_bin,
        c.died_within_icu_stay AS original_event_flag,

        -- Determine the final event/censoring time bin
        -- If the event happened after 33 days (bin 197), cap it at 197.
        CASE
            WHEN FLOOR(c.survival_time_hours / 4.0)::INTEGER > 197 THEN 197
            ELSE FLOOR(c.survival_time_hours / 4.0)::INTEGER
        END AS event_time_bin,

        -- Determine the final event flag
        -- The event (death) only counts if it happened *within* the 33-day window.
        -- All others are censored (flag=0).
        CASE
            WHEN c.died_within_icu_stay = 1 AND FLOOR(c.survival_time_hours / 4.0)::INTEGER <= 197 THEN 1
            ELSE 0
        END AS event_flag

    FROM
        {icu} ic
    JOIN
        {cohort} c ON ic.hadm_id = c.hadm_id;
    """
    ddb.execute(query)
    print(f"Created {output_table}.")
    save_parquet(ddb, out_dir="data/preprocessing_checkpoints", table=output_table, filename=f"{output_table.lower()}.parquet")

