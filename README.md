**Code to accompany,
    Stephen R. Pfohl, Yizhe Xu, Agata Foryciarz, Nikolaos Ignatiadis, Julian Genkins, and Nigam H. Shah. 2022. Net benefit, calibration, threshold selection, and training objectives for algorithmic fairness in healthcare. In 2022 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’22), June 21–24, 2022, Seoul, Republic of Korea. ACM, New York, NY, USA, 20 pages. https://doi.org/10.1145/3531146.3533166**

Available on Arxiv at: https://arxiv.org/abs/2202.01906

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