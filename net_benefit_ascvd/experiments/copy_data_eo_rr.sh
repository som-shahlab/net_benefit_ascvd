#!/bin/bash



DATA_PATH_OPTUM='/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod/'

TARGET_DIRECTORY_OPTUM='./figures_data/eo_rr'

mkdir -p $TARGET_DIRECTORY_OPTUM
cp \
    $DATA_PATH_OPTUM'/experiments/ascvd_10yr_optum_dod_regularized_eo_rr_selected/result_df_ci_race_eth.parquet' \
    $TARGET_DIRECTORY_OPTUM'/'

cp \
    $DATA_PATH_OPTUM'/experiments/ascvd_10yr_optum_dod_regularized_eo_rr_selected/result_df_ci_gender_concept_name.parquet' \
    $TARGET_DIRECTORY_OPTUM'/'

cp \
    $DATA_PATH_OPTUM'/experiments/ascvd_10yr_optum_dod_regularized_eo_rr_selected/result_df_ci_race_eth_gender.parquet' \
    $TARGET_DIRECTORY_OPTUM'/'


cp \
    $DATA_PATH_OPTUM'/experiments/ascvd_10yr_optum_dod_regularized_eo_rr_selected/result_df_ci_performance_race_eth.parquet' \
    $TARGET_DIRECTORY_OPTUM'/'

cp \
    $DATA_PATH_OPTUM'/experiments/ascvd_10yr_optum_dod_regularized_eo_rr_selected/result_df_ci_performance_gender_concept_name.parquet' \
    $TARGET_DIRECTORY_OPTUM'/'

cp \
    $DATA_PATH_OPTUM'/experiments/ascvd_10yr_optum_dod_regularized_eo_rr_selected/result_df_ci_performance_race_eth_gender.parquet' \
    $TARGET_DIRECTORY_OPTUM'/'