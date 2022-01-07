import os
import numpy as np
import pandas as pd

import argparse
import random

from net_benefit_ascvd.prediction_utils.pytorch_utils.metrics import RocNbEvaluator

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_path",
    type=str,
    default="",
    help="The root path where data is stored",
    required=True,
)

parser.add_argument("--seed", type=int, default=718)
parser.add_argument("--n_boot", type=int, default=1000)
parser.add_argument("--eval_attribute", type=str, required=True)
parser.add_argument("--result_path", type=str, required=True)
parser.add_argument("--result_filename", type=str, default="result_df_ci_eo")
parser.add_argument(
    "--n_jobs",
    type=int,
    default=None,
    help="The number of cores to use for bootstrapping",
)

parser.add_argument(
    "--perform_aggregation",
    dest="perform_aggregation",
    action="store_true",
    default=False,
)

parser.add_argument("--bootstrap_result_no_agg_load_path", default=None)


def prepare_output_df(output_df, phase=None):
    if phase is not None:
        output_df = output_df.query("phase == @phase")
    output_df_cohort = output_df.merge(cohort_df)
    output_df_cohort = output_df_cohort.query('training_fold_id=="@fold_id"')
    experiment_path_eo = os.path.join(
        data_path, "experiments", "ascvd_10yr_optum_dod_regularized_eo"
    )
    config_df_eo_path = os.path.join(experiment_path_eo, "config", "config.csv")
    config_df_eo = pd.read_csv(config_df_eo_path)
    config_df_eo["config_filename"] = config_df_eo.id.astype(str) + ".yaml"
    config_df_eo = config_df_eo.assign(
        experiment_path=experiment_path_eo,
        performance_dir="performance",
        filename="output_df.parquet",
        output_df_path=lambda x: x.experiment_path.str.cat(
            [x.performance_dir, x.config_filename, x.filename], sep=os.sep
        ),
    )
    selected_config_df_eo = config_df_eo
    output_df_eo = (
        pd.concat(
            {
                (
                    row.lambda_group_regularization,
                    row.group_objective_metric,
                    row.config_filename,
                    row.fold_id,
                ): pd.read_parquet(row.output_df_path)
                for i, row in selected_config_df_eo[
                    [
                        "output_df_path",
                        "lambda_group_regularization",
                        "group_objective_metric",
                        "config_filename",
                        "fold_id",
                    ]
                ]
                .drop_duplicates()
                .iterrows()
            }
        )
        .reset_index(level=-1, drop=True)
        .rename_axis(
            [
                "lambda_group_regularization",
                "group_objective_metric",
                "config_filename",
                "training_fold_id",
            ]
        )
        .reset_index()
        .assign(experiment_name="ascvd_10yr_optum_dod_regularized_eo")
    )
    if phase is not None:
        output_df_eo = output_df_eo.query("phase == @phase")

    output_df_combined = pd.concat(
        [
            output_df.assign(
                lambda_group_regularization=0.0, group_objective_metric=metric
            )
            for metric in selected_config_df_eo.group_objective_metric.unique()
        ]
    )
    output_df_combined = pd.concat(
        [
            output_df_combined,
            output_df_eo.drop(columns=["config_filename", "experiment_name"]),
        ]
    )
    output_df_cohort_combined = output_df_combined.merge(cohort_df)
    output_df_cohort_combined = output_df_cohort_combined.assign(
        training_fold_id=lambda x: x.training_fold_id.astype(int)
    )
    assert output_df_cohort_combined.shape[0] == output_df_combined.shape[0]
    return output_df_cohort_combined


if __name__ == "__main__":
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_path = args.data_path
    experiment_path = os.path.join(data_path, "experiments")
    selected_configs_path = os.path.join(
        experiment_path, "ascvd_10yr_optum_dod_selected_erm", "selected_configs.csv"
    )
    print(data_path)
    cohort_path = os.path.join(data_path, "cohort", "cohort_fold_1_5_ipcw.parquet")

    cohort_df = pd.read_parquet(cohort_path)
    selected_configs_df = pd.read_csv(selected_configs_path)
    selected_configs_df
    experiment_name = "ascvd_10yr_optum_dod_erm_tuning"
    selected_configs_df = selected_configs_df.assign(
        experiment_path=experiment_path,
        performance_dir="performance",
        filename="output_df.parquet",
        output_df_path=lambda x: x.experiment_path.str.cat(
            [x.experiment_name, x.performance_dir, x.config_filename, x.filename],
            sep=os.sep,
        ),
    )

    output_df = (
        pd.concat(
            {
                x["fold_id"]: pd.read_parquet(x["output_df_path"])
                for x in selected_configs_df.query(
                    "experiment_name == @experiment_name"
                ).to_dict("records")
            }
        )
        .reset_index(level=-1, drop=True)
        .rename_axis("training_fold_id")
        .reset_index()
        .assign(training_fold_id=lambda x: x.training_fold_id.astype(str))
    )
    output_df_cohort_combined = prepare_output_df(output_df, phase="test")
    output_df_cohort_combined = output_df_cohort_combined.assign(
        experiment=lambda x: x.group_objective_metric.str.cat(
            x.lambda_group_regularization.astype(str), sep="_"
        )
    )
    output_df_cohort_combined.loc[
        output_df_cohort_combined.lambda_group_regularization == 0.0, "experiment"
    ] = "ERM"

    temp_df = output_df_cohort_combined.query('phase == "test"')
    interpolation_values = np.linspace(0.001, 0.4, 500)
    evaluator = RocNbEvaluator()

    if args.bootstrap_result_no_agg_load_path is not None:
        bootstrap_result_no_agg = pd.read_parquet(
            args.bootstrap_result_no_agg_load_path
        )
    else:
        bootstrap_result_no_agg = None

    result_df = evaluator.bootstrap_roc_df2(
        temp_df,
        n_boot=args.n_boot,
        strata_vars_eval=[
            "phase",
            "training_fold_id",
            "experiment",
            "group_objective_metric",
            args.eval_attribute,
        ],
        metrics=[
            "fpr",
            "tpr",
            "fpr_implied",
            "tpr_implied",
            "calibration_density",
            "nb",
            "nb_all",
            "nb_none",
            "nb_implied",
            "nb_0.075",
            "nb_0.075_implied",
            "nb_0.2",
            "nb_0.2_implied",
        ],
        strata_vars_boot=["phase", "labels", args.eval_attribute],
        strata_var_replicate="training_fold_id",
        replicate_aggregation_mode=None,
        strata_var_experiment="experiment",
        baseline_experiment_name="ERM",
        weight_var="ipcw_weight",
        patient_id_var="row_id",
        n_jobs=args.n_jobs,
        verbose=True,
        nb_thresholds=[0.075, 0.2],
        interpolation_values=interpolation_values,
        perform_aggregation=args.perform_aggregation,
        bootstrap_result_no_agg=bootstrap_result_no_agg,
        rr=True,
        risk_reduction=0.275,
    )
    if args.perform_aggregation:
        result_df = result_df.assign(
            lambda_group_regularization=lambda x: x.apply(
                lambda y: float(y.experiment.split("_")[-1])
                if "_" in y.experiment
                else 0.0,
                axis=1,
            )
        )
        os.makedirs(args.result_path, exist_ok=True)
        result_df.to_csv(
            os.path.join(
                args.result_path,
                "{}_{}.csv".format(args.result_filename, args.eval_attribute),
            ),
            index=False,
        )
    else:
        os.makedirs(args.result_path, exist_ok=True)
        result_df.to_parquet(
            os.path.join(
                args.result_path,
                "{}_{}_no_agg.parquet".format(
                    args.result_filename, args.eval_attribute
                ),
            ),
            index=False,
        )
