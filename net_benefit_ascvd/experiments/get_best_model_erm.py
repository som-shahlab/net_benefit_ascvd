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
    required=True
)

parser.add_argument("--task_prefix", type=str, required=True)

parser.add_argument(
    "--selected_config_experiment_suffix",
    type=str,
    default="selected",
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

    ## Put it all together and export config files
    selected_config_df = pd.concat(
        [
            selected_config_df_erm
        ]
    )
    selected_config_df

    selected_config_experiment_name = "{}_{}".format(
        task_prefix, args.selected_config_experiment_suffix
    )
    selected_config_experiment_path = os.path.join(
        project_dir,
        "experiments",
        selected_config_experiment_name,
    )
    os.makedirs(selected_config_experiment_path, exist_ok=True)
    selected_config_df.to_csv(
        os.path.join(selected_config_experiment_path, "selected_configs.csv"),
        index=False,
    )
