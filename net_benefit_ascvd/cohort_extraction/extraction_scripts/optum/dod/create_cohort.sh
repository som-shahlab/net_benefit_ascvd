#!/bin/bash

GCLOUD_PROJECT='som-nero-phi-nigam-optum'
DATASET_PROJECT='som-rit-phi-starr-prod'
RS_DATASET_PROJECT='som-nero-phi-nigam-optum'
TEMP_DATASET_PROJECT='som-nero-phi-nigam-optum'

DATASET='optum_dod_2019q1_cdm_53'
RS_DATASET='ascvd_cohort_tables'
TEMP_DATASET='temp_dataset'

DB_KEY='optum_dod'
HORIZON='10yr'
COHORT_NAME="ascvd_"$HORIZON"_"$DATASET
COHORT_NAME_SAMPLED=$COHORT_NAME"_sampled"

python -m net_benefit_ascvd.cohort_extraction.create_cohort \
    --gcloud_project=$GCLOUD_PROJECT \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --temp_dataset_project=$TEMP_DATASET_PROJECT \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --temp_dataset=$TEMP_DATASET \
    --cohort_name=$COHORT_NAME \
    --cohort_name_sampled=$COHORT_NAME_SAMPLED \
    --db_key=$DB_KEY \
    --horizon=$HORIZON \
    --sample