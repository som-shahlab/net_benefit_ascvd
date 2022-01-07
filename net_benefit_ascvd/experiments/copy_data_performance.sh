#!/bin/bash



DATA_PATH_OPTUM='/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod/'

TARGET_DIRECTORY_OPTUM='./figures_data/performance'

mkdir -p $TARGET_DIRECTORY_OPTUM
cp \
    $DATA_PATH_OPTUM'experiments/ascvd_10yr_optum_dod_selected/result_df_ci.csv' \
    $TARGET_DIRECTORY_OPTUM'/'