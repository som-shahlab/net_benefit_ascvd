
PROJECT_DIR="/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod"
SELECTED_CONFIG_EXPERIMENT_SUFFIX="selected_erm"
TASK_PREFIX="ascvd_10yr_optum_dod"
python -m net_benefit_ascvd.experiments.get_best_model_erm \
    --project_dir=$PROJECT_DIR \
    --task_prefix=$TASK_PREFIX \
    --selected_config_experiment_suffix=$SELECTED_CONFIG_EXPERIMENT_SUFFIX
