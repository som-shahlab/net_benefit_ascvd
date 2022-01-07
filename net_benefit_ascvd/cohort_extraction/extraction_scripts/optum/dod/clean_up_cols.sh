#!/bin/bash

DATA_PATH="/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod/cohort"

python -m net_benefit_ascvd.cohort_extraction.clean_up_cols \
    --source_cohort_path=$DATA_PATH'/cohort_fold_1_5.parquet' \
    --target_cohort_path=$DATA_PATH'/cohort_fold_1_5.parquet'