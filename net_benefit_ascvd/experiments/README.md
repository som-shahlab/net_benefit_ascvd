#### Code to run experiments

##### Directories
    1. `cohort_extraction` (this directory): Contains .py files that define the experimental pipeline
    2. `extraction_scripts`: Directory containing `.sh` scripts that call `.py` files to execute the experiments

##### Files
    * `experiments`
        * `derive_ipcw_weights`: Code to derive IPCW weights
        * `create_grids.py`: Creates yaml config files for tuning grid
        * `train_model.py`: Script that calls model training procedure
        * `run_tuning.py`: Script that can be used to parallelize job execution
        * `get_best_model_erm.py`: Script that performs model selection for ERM models
        * `get_best_model_all.py`: Script that performance model selection for all other models
        * `bootstrapping_performance`: Script that generates bootstrap results for the `performance` experiments
        * `bootstrapping_eo_rr`: Script that generates bootstrap results for the equalized odds experiments at a range of thresholds/scores
        * `bootstrapping_eo_rr_performance`: Script that generates aggregate bootstrap results for the equalized odds experiments (not across a range of thresholds/scores)
        * `copy_data_*.sh`: Script that copies bootstrap results to `figures_data/`
        * `util.py`: Defines utility functions
    * `scripts`
        * See README within for description of individual files and execution order