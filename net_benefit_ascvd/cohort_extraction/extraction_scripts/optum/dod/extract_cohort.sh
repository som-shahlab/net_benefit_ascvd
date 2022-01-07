#!/bin/bash

GCLOUD_PROJECT='som-nero-phi-nigam-optum'
RS_DATASET_PROJECT='som-nero-phi-nigam-optum'
DATASET='optum_dod_2019q1_cdm_53'
RS_DATASET='ascvd_cohort_tables'
HORIZON='10yr'
COHORT_NAME="ascvd_"$HORIZON"_"$DATASET"_sampled"
DATA_PATH="/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod"

python -m net_benefit_ascvd.cohort_extraction.extract_cohort \
    --data_path=$DATA_PATH \
    --gcloud_project=$GCLOUD_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --rs_dataset=$RS_DATASET \
    --cohort_name=$COHORT_NAME