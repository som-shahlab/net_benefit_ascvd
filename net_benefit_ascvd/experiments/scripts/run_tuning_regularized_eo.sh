#!/bin/bash

DATA_PATH_PREFIX="/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod"

train_model_func() {
    python -m net_benefit_ascvd.experiments.run_tuning \
        --data_path=$DATA_PATH_PREFIX \
        --experiment_name=$1'_regularized_eo' \
        --cuda_device=$2 \
        --max_concurrent_jobs=10 \
        --num_workers=0 \
        --cohort_name="cohort_fold_1_5_ipcw.parquet" \
        --weight_var_name='ipcw_weight'
}

train_model_func 'ascvd_10yr_optum_dod' 3