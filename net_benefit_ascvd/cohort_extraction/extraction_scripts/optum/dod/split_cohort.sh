#!/bin/bash

DATA_PATH="/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod/cohort"
FEATURES_ROW_ID_MAP_PATH="/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod/merged_features_binary/features_sparse/features_row_id_map.parquet"
python -m net_benefit_ascvd.cohort_extraction.split_cohort \
    --source_cohort_path=$DATA_PATH'/cohort.parquet' \
    --target_cohort_path=$DATA_PATH'/cohort_fold_1_5.parquet' \
    --features_row_id_map_path=$FEATURES_ROW_ID_MAP_PATH \
    --test_frac=0.25 \
    --nfold=5 \
    --seed=5573