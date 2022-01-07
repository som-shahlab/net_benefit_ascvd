

CUDA_VISIBLE_DEVICES=3 \
    /local-scratch/nigam/envs/prediction_utils/bin/python -m net_benefit_ascvd.experiments.derive_ipcw_weights \
    --result_file_name="cohort_fold_1_5_ipcw.parquet" \
    --num_hidden=1 \
    --db_key='optum_dod'