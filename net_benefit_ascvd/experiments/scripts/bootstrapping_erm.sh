#!/bin/bash 


N_BOOT=1000
SEED=718
TASK="ascvd_binary"
WEIGHT_VAR_NAME="ipcw_weight"

DATA_PATH="/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod"
COHORT_PATH=$DATA_PATH"/cohort/cohort_fold_1_5_ipcw.parquet"

SELECTED_CONFIG_EXPERIMENT_NAME="ascvd_10yr_optum_dod_selected_erm" 
BASELINE_EXPERIMENT_NAME="ascvd_10yr_optum_dod_erm_tuning"

python -m net_benefit_ascvd.experiments.bootstrapping_performance \
    --data_path=$DATA_PATH \
    --cohort_path=$COHORT_PATH \
    --n_boot=$N_BOOT \
    --seed=$SEED \
    --task=$TASK \
    --selected_config_experiment_name=$SELECTED_CONFIG_EXPERIMENT_NAME \
    --baseline_experiment_name=$BASELINE_EXPERIMENT_NAME \
    --weight_var_name=$WEIGHT_VAR_NAME \
    --metrics "auc" "loss_bce" "ace_abs_logistic_logit" \
    --n_jobs=8 \
    --eval_attributes "race_eth" "gender_concept_name" "age_group" "race_eth_gender" "has_ra_history" "has_ckd_history" "has_diabetes_type2_history" "has_diabetes_type1_history"