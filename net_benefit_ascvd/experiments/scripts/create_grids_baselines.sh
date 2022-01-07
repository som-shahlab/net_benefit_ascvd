
DATA_PATH="/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod"
COHORT_PATH=$DATA_PATH'/cohort/cohort_fold_1_5_ipcw.parquet'
EXPERIMENT_NAME='ascvd_10yr_optum_dod'

# (TODO) Consider removing gender_concept_name and age_group

python -m net_benefit_ascvd.experiments.create_grids \
    --data_path=$DATA_PATH \
    --cohort_path=$COHORT_PATH \
    --experiment_name_prefix=$EXPERIMENT_NAME \
    --tasks 'ascvd_binary' \
    --attributes "race_eth" "gender_concept_name" "age_group" "race_eth_gender" "has_ra_history" "has_ckd_history" "has_diabetes_type2_history" "has_diabetes_type1_history" \
    --num_folds=5 \
    --experiment_names "erm_tuning" "erm_subset_tuning"