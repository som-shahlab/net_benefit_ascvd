#!/bin/bash 


N_BOOT=1000
SEED=718

DATA_PATH="/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod"
RESULT_PATH=$DATA_PATH"/experiments/ascvd_10yr_optum_dod_regularized_eo_rr_selected"
EVAL_ATTRIBUTES=("race_eth" "gender_concept_name" "race_eth_gender")

for eval_attribute in "${EVAL_ATTRIBUTES[@]}"
do
    echo $eval_attribute
    python -m net_benefit_ascvd.experiments.bootstrapping_eo_rr_performance \
        --data_path=$DATA_PATH \
        --n_boot=$N_BOOT \
        --seed=$SEED \
        --n_jobs=8 \
        --eval_attribute=$eval_attribute \
        --result_path=$RESULT_PATH \
        --result_filename="result_df_ci_performance"
done