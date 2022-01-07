#!/bin/bash


GCLOUD_PROJECT='som-nero-phi-nigam-optum'
DATASET_PROJECT='som-rit-phi-starr-prod'
RS_DATASET_PROJECT='som-nero-phi-nigam-optum'

DATASET='optum_dod_2019q1_cdm_53'
RS_DATASET='ascvd_cohort_tables'

DATA_PATH="/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod"

COHORT_NAME="ascvd_10yr_"$DATASET"_sampled"

FEATURES_DATASET="temp_dataset"
FEATURES_PREFIX="features_"$USER

INDEX_DATE_FIELD='index_date'
ROW_ID_FIELD='prediction_id'
MERGED_NAME='merged_features_binary'

python -m net_benefit_ascvd.prediction_utils.extraction_utils.extract_features \
    --data_path=$DATA_PATH \
    --features_by_analysis_path="features_by_analysis" \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --cohort_name=$COHORT_NAME \
    --gcloud_project=$GCLOUD_PROJECT \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --features_dataset=$FEATURES_DATASET \
    --features_prefix=$FEATURES_PREFIX \
    --index_date_field=$INDEX_DATE_FIELD \
    --row_id_field=$ROW_ID_FIELD \
    --merged_name=$MERGED_NAME \
    --analysis_ids \
        "condition_occurrence_delayed" \
        "procedure_occurrence_delayed" \
        "drug_exposure" \
        "device_exposure" \
        "measurement" \
        "note_type_delayed" \
        "observation" \
        "measurement_range" \
        "gender" \
        "race" \
        "ethnicity" \
        "age_group" \
    --time_bins -365 -30 0 \
    --binary \
    --featurize \
    --no_cloud_storage \
    --merge_features \
    --create_sparse \
    --no_create_parquet \
    --overwrite