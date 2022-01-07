import numpy as np
import os
import pandas as pd
import configargparse as argparse

from net_benefit_ascvd.prediction_utils.util import yaml_write
from net_benefit_ascvd.experiments.train_model import filter_cohort
from net_benefit_ascvd.experiments.util import generate_grid

parser = argparse.ArgumentParser(
    config_file_parser_class=argparse.YAMLConfigFileParser,
)

parser.add_argument(
    "--data_path", type=str, default="", help="The root data path",
)

parser.add_argument(
    "--cohort_path",
    type=str,
    default="",
    help="File name for the file containing metadata",
)

parser.add_argument(
    "--experiment_name_prefix",
    type=str,
    default="scratch",
    help="The name of the experiment",
)


parser.add_argument(
    "--grid_size",
    type=int,
    default=None,
    help="The number of elements in the random grid",
)

parser.add_argument(
    "--attributes",
    type=str,
    nargs="*",
    required=False,
    default=[
        "race_eth",
        "gender_concept_name",
        "age_group",
        "race_eth_gender",
        "has_ra_history",
        "has_ckd_history",
        "has_diabetes_type2_history",
        "has_diabetes_type1_history",
    ],
)

parser.add_argument(
    "--tasks", type=str, nargs="*", required=False, default=["ascvd_binary"],
)

parser.add_argument(
    "--num_folds", type=int, required=False, default=5,
)

parser.add_argument("--seed", type=int, default=234)

parser.add_argument("--experiment_names", nargs="*", required=False, default=None)


if __name__ == "__main__":

    args = parser.parse_args()
    cohort = pd.read_parquet(args.cohort_path)
    cohort = filter_cohort(cohort)

    common_grid = {
        "erm": {
            "global": {
                "lr": [1e-4, 1e-5],
                "batch_size": [512],
                "num_epochs": [150],
                "gamma": [1.0],
                "early_stopping": [True],
                "early_stopping_patience": [25],
            },
            "model_specific": {
                "feedforward_net": {
                    "drop_prob": [0.25, 0.75],
                    "num_hidden": [1, 3],
                    "hidden_dim": [128, 256],
                },
                "logistic_regression": {
                    "num_hidden": [0],
                    "weight_decay": [0, 1e-2, 1e-1],
                },
            },
            "experiment": {
                "label_col": args.tasks,
                "fold_id": [str(i + 1) for i in range(args.num_folds)],
            },
        },
        "regularized_eo": {
            "global": {
                "lr": [1e-4],
                "batch_size": [512],
                "num_epochs": [150],
                "gamma": [1.0],
                "early_stopping": [True],
                "early_stopping_patience": [25],
                "thresholds": [[0.075, 0.2]],
                "threshold_mode": ["conditional"],
                "mmd_mode": ["conditional"]
            },
            "model_specific": {
                "feedforward_net": {
                    "drop_prob": [0.25],
                    "num_hidden": [3],
                    "hidden_dim": [256],
                }
            },
            "experiment": {
                "label_col": args.tasks,
                "fold_id": [str(i + 1) for i in range(args.num_folds)],
                "lambda_group_regularization": [
                    float(x) for x in np.logspace(-2, 1, num=5)
                ],
                "group_objective_metric": ["mmd", "threshold_rate"],
                "sensitive_attribute": args.attributes,
            },
        },
        "regularized_performance": {
            "global": {
                "lr": [1e-4],
                "batch_size": [512],
                "num_epochs": [150],
                "gamma": [1.0],
                "early_stopping": [True],
                "early_stopping_patience": [25],
            },
            "model_specific": {
                "feedforward_net": {
                    "drop_prob": [0.25],
                    "num_hidden": [3],
                    "hidden_dim": [256],
                }
            },
            "experiment": {
                "label_col": args.tasks,
                "fold_id": [str(i + 1) for i in range(args.num_folds)],
                "lambda_group_regularization": [
                    float(x) for x in np.logspace(-2, 1, num=5)
                ],
                "group_objective_metric": ["auc", "loss"],
                "sensitive_attribute": args.attributes,
            },
        },
        "dro": {
            "global": {
                "lr": [1e-4],
                "batch_size": [512],
                "num_epochs": [150],
                "gamma": [1.0],
                "early_stopping": [True],
                "early_stopping_patience": [25],
                "drop_prob": [0.25],
                "num_hidden": [3],
                "hidden_dim": [256],
                "lr_lambda": [1, 1e-1, 1e-2],
            },
            "model_specific": {
                "auc_selection": {
                    "selection_metric": ["auc_min"],
                    "group_objective_metric": ["auc_proxy"],
                },
                "loss_selection": {
                    "selection_metric": ["loss_bce_max"],
                    "group_objective_metric": ["loss"],
                },
            },
            "experiment": {
                "label_col": args.tasks,
                "fold_id": [str(i + 1) for i in range(args.num_folds)],
            },
        },
    }

    grids = {
        "erm_tuning": {
            "global": common_grid["erm"]["global"],
            "model_specific": [
                common_grid["erm"]["model_specific"]["feedforward_net"],
                common_grid["erm"]["model_specific"]["logistic_regression"],
            ],
            "experiment": {
                **common_grid["erm"]["experiment"],
                **{
                    "group_objective_type": ["standard"],
                    "balance_groups": [False],
                    "selection_metric": ["loss"],
                },
            },
        },
        "erm_subset_tuning": {
            "global": common_grid["erm"]["global"],
            "model_specific": [
                common_grid["erm"]["model_specific"]["feedforward_net"],
                common_grid["erm"]["model_specific"]["logistic_regression"],
            ],
            "experiment": [
                {
                    **common_grid["erm"]["experiment"],
                    **{"group_objective_type": ["standard"]},
                    "subset_attribute": [attribute],
                    "subset_group": list(cohort[attribute].unique()),
                }
                for attribute in args.attributes
            ],
        },
        "regularized_eo": {
            "global": common_grid["regularized_eo"]["global"],
            "model_specific": [
                common_grid["regularized_eo"]["model_specific"]["feedforward_net"]
            ],
            "experiment": {
                **common_grid["regularized_eo"]["experiment"],
                **{
                    "group_objective_type": ["regularized"],
                    "balance_groups": [False],
                    "selection_metric": ["loss"],
                    "thresholds": [[0.075, 0.2]],
                },
            },
        },
        "regularized_performance": {
            "global": common_grid["regularized_performance"]["global"],
            "model_specific": [
                common_grid["regularized_performance"]["model_specific"][
                    "feedforward_net"
                ]
            ],
            "experiment": {
                **common_grid["regularized_performance"]["experiment"],
                **{
                    "group_objective_type": ["regularized"],
                    "balance_groups": [False],
                    "selection_metric": ["loss"],
                },
            },
        },
        "dro": {
            "global": {**common_grid['dro']["global"]},
            "model_specific": [
                common_grid['dro']["model_specific"]["auc_selection"],
                common_grid['dro']["model_specific"]["loss_selection"],
            ],
            "experiment": {
                **common_grid['dro']["experiment"],
                **{
                    "group_objective_type": ["dro"],
                    "sensitive_attribute": args.attributes,
                    "balance_groups": [True, False],
                },
            },
        },
    }

    if args.experiment_names is None:
        experiment_names = grids.keys()
    else:
        experiment_names = args.experiment_names

    for experiment_name in experiment_names:
        print(experiment_name)
        the_grid = generate_grid(
            grids[experiment_name]["global"],
            grids[experiment_name]["model_specific"],
            grids[experiment_name]["experiment"],
        )
        print("{}, length: {}".format(experiment_name, len(the_grid)))

        experiment_dir_name = "{}_{}".format(
            args.experiment_name_prefix, experiment_name
        )

        grid_df = pd.DataFrame(the_grid)
        config_path = os.path.join(
            args.data_path, "experiments", experiment_dir_name, "config"
        )
        os.makedirs(config_path, exist_ok=True)
        grid_df.to_csv(os.path.join(config_path, "config.csv"), index_label="id")

        for i, config_dict in enumerate(the_grid):
            yaml_write(config_dict, os.path.join(config_path, "{}.yaml".format(i)))
