**Code to accompany,
    "Net benefit, calibration, threshold selection, and training objectives for algorithmic fairness in healthcare" by Stephen R. Pfohl, Yizhe Xu, Agata Foryciarz, Nikolaos Ignatiadis, Nigam H. Shah**

## Installation
1. Pull this repository, and from within the project directory call `pip install .` or `pip install -e .` if you intend to modify the code

## Package structure
* The package is structured into the following submodules. Further information on each submodule is provided README files within the respective directories. 
* `net_benefit_ascvd.cohort_extraction`
    * Contains code to define the cohort and extract the data from the database
* `net_benefit_ascvd.experiments`
    * Contains code to execute the experiments
* `net_benefit_ascvd.prediction_utils`
    * Contains library code defining models and procedures for training and evaluation
* `notebooks`
    * Contains notebooks for data visualization and analysis
* Code to generate supplementary simulation results can be found at: https://github.com/spfohl/fairness_simulation