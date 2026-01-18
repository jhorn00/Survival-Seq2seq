# Preprocessing

**Important:** You MUST obtain a copy of MIMIC-IV 1.0 on your own to run the preprocessing or training code. We would be in violation of our data access agreements if we were to share any of the MIMIC data ourselves.

## Usage Instructions
- Configure and activate your `conda` environment if you have not already. (See `Survival-Seq2seq/README.md/Getting Started`)
- Change the constant `PATH_TO_RAW_MIMIC_DATA` at the top of `pipeline.py` to match where your raw MIMIC-IV 1.0 data is located
- Run preprocessing pipeline with `python pipeline.py`

## Background and Notes
### Data filters and restrictions
#### 24 Hour Window
Using a 24-hour window from the first instance of ICU data is a common restriction with a long precedent in academic works. It is highly likely the same filter was applied to the data in the paper we are attempting to replicate.
- Most notably, the [MIMIC Code Repository](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/firstday) and [MIMIC-Extract Pipeline](https://github.com/MLforHealth/MIMIC_Extract) use this restriction for ICU data
- These apparently follow the lead of scoring methodologies such as [SAPS-II](https://pubmed.ncbi.nlm.nih.gov/8254858/) and [APACHE II (The linked paper uses APACHE II and summarizes its methodology)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10060092/)
- [OASIS also makes use of information within the first 24 hours of ICU admission](https://pubmed.ncbi.nlm.nih.gov/23660729/)
This can help avoid correlation stemming from readmission.

#### Adult Patients
It is common to restict to adult patients due to the complexities in modeling for youth and adults, however the dataset already has this restriction present in the `ICUSTAYS` data.

### Feature Selection
1. Top Numeric Features in CHARTEVENTS (most common)
2. Top Numeric Features in INPUTEVENTS (most common)
3. Top Numeric Features in OUTPUTEVENTS (most common)
4. Select subset of top features
    - Aggregation types noted
5. Generate the 4 hour bins for ICUSTAYS - We may want to reduce this set before the actual pipeline gets run
    - 4 hour blocks with partial data are kept
6. Aggregate features from CHARTEVENTS
    - Fetch the ids of the features (placed in list `ce_avg_ids`)
    - Fetch the ids and their values for a pivot list (`key AS value`, or for example `220045 AS HeartRate`)
    - Subquery to determine which time bin the event falls into
        - This needs a join on ICUSTAYS to figure out what 4hr bin of the stay the event datetime translates to
        - This relies on `valuenum` existing in the CHARTEVENTS entry (not null). We would have to account for this when dealing with non-numerics.
        - Resulting rows are `(stay_id, time_bin_index, itemid, valuenum)`
    - PIVOT:
        - For each itemid in the pivot list (as the value string, so the column becomes that string) group the stay_id and time_bin_index and aggregate the valuenum over it
            - In other words, for Heart rate it would average the heart rate for every 4 hour block
            - This is done for each of the itemids (metrics like HR, RR, etc.)
7. Handle NULL entries in CHARTEVENTS
    - Replace with 0
    - Flag if it was NULL before
8. Aggregate features from OUTPUTEVENTS but with SUM rather than AVG
9. Handle NULL entries in OUTPUTEVENTS
    - Replace with 0
    - Flag if it was NULL before
10. Aggregate features from INPUTEVENTS but with SUM rather than AVG
11. Handle NULL entries in INPUTEVENTS
    - Replace with 0
    - Flag if it was NULL before

###### Old NULL Handling
- Forward fill
- Otherwise backward fill
- Otherwise fill with average-case (This covers if they never had a measurement)
- Flag if it was NULL before

###### New NULL Handling
- Replace with 0
- Flag if it was NULL before
