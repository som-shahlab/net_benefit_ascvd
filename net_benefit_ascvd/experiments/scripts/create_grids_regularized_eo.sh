
DATA_PATH="/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod"
COHORT_PATH=$DATA_PATH'/cohort/cohort_fold_1_5_ipcw.parquet'
EXPERIMENT_NAME='ascvd_10yr_optum_dod'

python -m net_benefit_ascvd.experiments.create_grids \
    --data_path=$DATA_PATH \
    --cohort_path=$COHORT_PATH \
    --experiment_name_prefix=$EXPERIMENT_NAME \
    --tasks 'ascvd_binary' \
    --attributes "race_eth_gender" \
    --num_folds=5 \
    --experiment_names "regularized_eo"