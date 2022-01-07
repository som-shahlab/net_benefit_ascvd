import pandas as pd
import glob
import os
import configargparse as argparse
from net_benefit_ascvd.prediction_utils.util import df_dict_concat


parser = argparse.ArgumentParser(
    config_file_parser_class=argparse.YAMLConfigFileParser,
)

parser.add_argument(
    "--project_dir",
    type=str,
)

parser.add_argument("--task_prefix", type=str, required=True)

parser.add_argument(
    "--selected_config_experiment_suffix", type=str, default="selected",
)

if __name__ == "__main__":

    args = parser.parse_args()
    project_dir = args.project_dir
    task_prefix = args.task_prefix

    def get_config_df(experiment_name):
        config_df_path = os.path.join(
            os.path.join(
                project_dir, "experiments", experiment_name, "config", "config.csv"
            )
        )
        config_df = pd.read_csv(config_df_path)
        config_df["config_filename"] = config_df.id.astype(str) + ".yaml"
        return config_df

    def get_result_df(
        experiment_name, output_filename="result_df_group_standard_eval.parquet"
    ):
        baseline_files = glob.glob(
            os.path.join(
                project_dir, "experiments", experiment_name, "**", output_filename
            ),
            recursive=True,
        )
        assert len(baseline_files) > 0
        baseline_df_dict = {
            tuple(file_name.split("/"))[-2]: pd.read_parquet(file_name)
            for file_name in baseline_files
        }
        baseline_df = df_dict_concat(baseline_df_dict, ["config_filename"])
        return baseline_df

    def append_hparams_id_col(config_df):
        config_df = (
            config_df[
                set(config_df.columns) - set(["id", "config_filename", "fold_id"])
            ]
            .drop_duplicates(ignore_index=True)
            .rename_axis("hparams_id")
            .reset_index()
            .merge(config_df)
        )
        return config_df

    def load_dataframes(
        experiment_name, output_filename="result_df_group_standard_eval.parquet"
    ):
        config_df = get_config_df(experiment_name)
        config_df = append_hparams_id_col(config_df)
        result_df = get_result_df(experiment_name, output_filename=output_filename)
        result_df = result_df.merge(config_df[["config_filename", "hparams_id"]])

        return result_df, config_df

    def query_df(df, query_str=None):
        if query_str is None:
            return df
        else:
            return df.query(query_str)

    def select_model_mean_min_max(
        result_df,
        config_df,
        metric_name="auc",
        agg_func_inner="min",
        agg_func_outer="max",
        query_str=None,
        group_vars=None,
    ):
        default_group_vars = ["sensitive_attribute", "eval_attribute"]
        group_vars = (
            default_group_vars
            if group_vars is None
            else default_group_vars + group_vars
        )
        mean_performance_by_hparam = (
            result_df.merge(config_df)
            .pipe(query_df, query_str=query_str)
            .query("sensitive_attribute == eval_attribute")
            .query("metric == @metric_name")
            .query('phase == "eval"')
            .groupby(group_vars + ["config_filename", "hparams_id"])
            .agg(performance=("performance", agg_func_inner))
            .reset_index()
            .groupby(group_vars + ["hparams_id"])
            .agg(performance=("performance", "mean"))
            .reset_index()
        )

        # Get the hparam_id with the best mean performance
        return (
            mean_performance_by_hparam.groupby(group_vars)
            .agg(performance=("performance", agg_func_outer))
            .merge(mean_performance_by_hparam)
        )

    experiment_names = {
        "erm_baseline": "{}_erm_tuning".format(task_prefix),
        "erm_subset": "{}_erm_subset_tuning".format(task_prefix),
        "dro": "{}_dro".format(task_prefix),
        "regularized_performance": "{}_regularized_performance".format(task_prefix),
    }

    ## ERM
    result_df_erm, config_df_erm = load_dataframes(
        experiment_names["erm_baseline"],
        output_filename="result_df_training_eval.parquet",
    )
    mean_performance = pd.DataFrame(
        result_df_erm.merge(config_df_erm)
        .query('metric == "loss_bce" & phase == "eval"')
        .groupby(["hparams_id"])
        .agg(performance=("performance", "mean"))
        .reset_index()
    )

    best_model = mean_performance.agg(performance=("performance", "min")).merge(
        mean_performance
    )
    selected_config_df_erm = (
        best_model[["hparams_id"]]
        .merge(config_df_erm)
        .assign(tag="erm_baseline", experiment_name=experiment_names["erm_baseline"])
    )
    erm_results = best_model[["hparams_id"]].merge(result_df_erm)

    ## ERM Subset
    result_df_subset, config_df_subset = load_dataframes(experiment_names["erm_subset"])
    mean_performance_subset = (
        result_df_subset.merge(
            config_df_subset[["config_filename", "subset_attribute", "subset_group"]]
        )
        .query("subset_attribute == eval_attribute & subset_group == subset_group")
        .query('metric == "loss_bce" & phase == "eval"')
        .groupby(["hparams_id", "subset_attribute", "subset_group"])
        .agg(performance=("performance", "mean"))
        .reset_index()
    )

    best_performance_subset = (
        mean_performance_subset.groupby(["subset_attribute", "subset_group"])
        .agg(performance=("performance", "min"))
        .reset_index()
        .merge(mean_performance_subset)
    )

    selected_models_subset = (
        best_performance_subset.merge(config_df_subset)
        .drop(columns="performance")
        .assign(tag="erm_subset", experiment_name=experiment_names["erm_subset"])
    )

    ## DRO
    result_df_dro, config_df_dro = load_dataframes(experiment_names["dro"])
    best_mean_auc_min_dro = select_model_mean_min_max(
        result_df_dro,
        config_df_dro.query('group_objective_metric == "auc_proxy"'),
        metric_name="auc",
        agg_func_inner="min",
        agg_func_outer="max",
    )
    best_mean_auc_min_dro = best_mean_auc_min_dro.assign(
        config_selection="auc_min_max", tag="dro_auc_min"
    )
    best_mean_loss_max_dro = select_model_mean_min_max(
        result_df_dro,
        config_df_dro.query('group_objective_metric == "loss"'),
        metric_name="loss_bce",
        agg_func_inner="max",
        agg_func_outer="min",
    )
    best_mean_loss_max_dro = best_mean_loss_max_dro.assign(
        config_selection="loss_max_min", tag="dro_loss_max"
    )
    selected_models_dro = (
        pd.concat([best_mean_auc_min_dro, best_mean_loss_max_dro])
        .drop(columns=["performance"])
        .merge(config_df_dro)
        .assign(experiment_name=experiment_names["dro"])
    )
    selected_models_dro

    # Regularized
    result_df_regularized, config_df_regularized = load_dataframes(
        experiment_names["regularized_performance"]
    )
    best_mean_auc_min_regularized = select_model_mean_min_max(
        result_df_regularized,
        config_df_regularized.query('group_objective_metric == "auc"'),
        metric_name="auc",
        agg_func_inner="min",
        agg_func_outer="max",
    )
    best_mean_auc_min_regularized = best_mean_auc_min_regularized.assign(
        config_selection="auc_min_max", tag="regularized_auc_min"
    )
    best_mean_loss_max_regularized = select_model_mean_min_max(
        result_df_regularized,
        config_df_regularized.query('group_objective_metric == "loss"'),
        metric_name="loss_bce",
        agg_func_inner="max",
        agg_func_outer="min",
    )
    best_mean_loss_max_regularized = best_mean_loss_max_regularized.assign(
        config_selection="loss_max_min", tag="regularized_loss_max"
    )
    selected_models_regularized = (
        pd.concat([best_mean_auc_min_regularized, best_mean_loss_max_regularized])
        .drop(columns=["performance"])
        .merge(config_df_regularized)
        .assign(experiment_name=experiment_names["regularized_performance"])
    )

    ## Put it all together and export config files
    selected_config_df = pd.concat(
        [
            selected_config_df_erm,
            selected_models_subset,
            selected_models_dro,
            selected_models_regularized,
        ]
    )
    selected_config_df

    selected_config_experiment_name = "{}_{}".format(
        task_prefix, args.selected_config_experiment_suffix
    )
    selected_config_experiment_path = os.path.join(
        project_dir, "experiments", selected_config_experiment_name,
    )
    os.makedirs(selected_config_experiment_path, exist_ok=True)
    selected_config_df.to_csv(
        os.path.join(selected_config_experiment_path, "selected_configs.csv"),
        index=False,
    )
