## Scripts to execute experiments

* Derive IPCW weights
    * `derive_ipcw_weights.sh`
* Create grids
    * `create_grids_baselines.sh`
    * `create_grids_dro.sh`
    * `create_grids_regularized_performance.sh`
    * `create_grids_regularized_eo.sh`
* Run main tuning loop
    * `run_tuning.sh`: baselines
    * `run_tuning_reg_dro.sh`: runs the following scripts
        * `run_tuning_dro.sh`
        * `run_tuning_regularized_performance.sh`
        * `run_tuning_regularized_eo.sh`
* Perform model selection
    * `get_best_model_erm.sh`
    * `get_best_model_performance.sh`
* Generate bootstrapping results
    * `bootstrapping_erm.sh`
    * `bootstrapping_performance.sh`
    * `bootstrapping_eo_rr_performance.sh`
    * `bootstrapping_eo_rr.sh`