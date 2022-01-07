#### Code to extract cohorts

##### Directories
    1. `cohort_extraction` (this directory): Contains .py files that define the extraction pipeline
    2. `extraction_scripts`: Directory containing `.sh` scripts that call `.py` files to execute the extraction

##### Files
    * `cohort_extraction`
        * `cohort.py`: Mixed python and SQL code that defines the cohort
        * `create_cohort.py`: Code that executes the code defined in cohort.py to create cohorts in BigQuery
        * `extract_cohort.py`: Pulls cohort table out of BigQuery and saves to file
        * `split_cohort.py`: Partitions the data into training, development, validation, and testing folds
        * `util.py`: Defines utility functions
    * `extraction_scripts`
        * `optum/dod`: Sub-directory
            * `run_extraction_0.sh` && `run_extraction_1.sh`: Runs extraction for the Optum cohort