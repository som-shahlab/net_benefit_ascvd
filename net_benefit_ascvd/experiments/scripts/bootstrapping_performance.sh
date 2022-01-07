#!/bin/bash 


N_BOOT=1000
SEED=718
TASK="ascvd_binary"
WEIGHT_VAR_NAME="ipcw_weight"

DATA_PATH="/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod"
COHORT_PATH=$DATA_PATH"/cohort/cohort_fold_1_5_ipcw.parquet"


SELECTED_CONFIG_EXPERIMENT_NAME="ascvd_10yr_optum_dod_selected" 
# BASELINE_EXPERIMENT_NAME="ascvd_10yr_optum_dod_erm_tuning"
EXPERIMENT_NAME_PREFIX="ascvd_10yr_optum_dod"
BASELINE_EXPERIMENT_NAME=$EXPERIMENT_NAME_PREFIX'_erm_tuning'
EXPERIMENT_NAME_SUFFIXES=('erm_tuning' 'erm_subset_tuning' 'dro' 'regularized_performance')

EVAL_ATTRIBUTES=("race_eth" "gender_concept_name" "race_eth_gender" "has_ra_history" "has_ckd_history" "has_diabetes_type2_history" "has_diabetes_type1_history")

for experiment_name_suffix in "${EXPERIMENT_NAME_SUFFIXES[@]}"
do
    echo $experiment_name_suffix
    for eval_attribute in "${EVAL_ATTRIBUTES[@]}"
    do
        echo $eval_attribute
        python -m net_benefit_ascvd.experiments.bootstrapping_performance \
            --data_path=$DATA_PATH \
            --cohort_path=$COHORT_PATH \
            --n_boot=$N_BOOT \
            --seed=$SEED \
            --task=$TASK \
            --selected_config_experiment_name=$SELECTED_CONFIG_EXPERIMENT_NAME \
            --baseline_experiment_name=$BASELINE_EXPERIMENT_NAME \
            --comparator_experiment_name=$EXPERIMENT_NAME_PREFIX"_"$experiment_name_suffix \
            --weight_var_name=$WEIGHT_VAR_NAME \
            --metrics "auc" "loss_bce" "ace_abs_logistic_logit" \
            --threshold_metrics 'net_benefit_rr' 'net_benefit_rr_recalib' \
            --thresholds 0.075 0.2 \
            --n_jobs=12 \
            --eval_attributes $eval_attribute \
            --result_filename="result_df_ci_"$experiment_name_suffix"_"$eval_attribute
    done
done