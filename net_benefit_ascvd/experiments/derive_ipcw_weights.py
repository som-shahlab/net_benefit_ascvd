import pandas as pd
import os
import joblib
import configargparse as argparse
from net_benefit_ascvd.prediction_utils.pytorch_utils.survival import (
    DiscreteTimeArrayLoaderGenerator,
    DiscreteTimeNNet,
)

parser = argparse.ArgumentParser(
    config_file_parser_class=argparse.YAMLConfigFileParser,
)

parser.add_argument(
    "--result_file_name", type=str, default="cohort_fold_1_5_ipcw.parquet"
)
parser.add_argument("--num_hidden", type=int, default=1)
parser.add_argument("--db_key", type=str, default="optum")


def fit_and_eval_discrete_time_model(
    features, cohort, time_horizon, censoring_model=True, **config_dict
):
    cohort = cohort.copy()
    if censoring_model:
        cohort["event_indicator"] = 1 - cohort["event_indicator"]
    loader_generator = DiscreteTimeArrayLoaderGenerator(
        features=features,
        cohort=cohort,
        **config_dict,
    )
    loaders = loader_generator.init_loaders()
    bin_transformer = loader_generator.config_dict["bin_transformer"]
    config_dict["output_dim"] = len(bin_transformer.kw_args["bins"]) - 1
    model = DiscreteTimeNNet(transformer=bin_transformer, **config_dict)
    model.train(loaders, **config_dict)
    output_dict = model.predict(
        loader_generator.init_loaders_predict(),
        phases=["val", "eval", "test"],
        time_horizon=time_horizon,
    )
    return output_dict["outputs"]


if __name__ == "__main__":
    args = parser.parse_args()
    data_paths = {
        "optum_dod": "/local-scratch/nigam/secure/optum/spfohl/zipcode_cvd/optum/dod",
    }

    cohort_paths = {
        key: os.path.join(value, "cohort", "cohort_fold_1_5.parquet")
        for key, value in data_paths.items()
    }

    features_paths = {
        key: os.path.join(
            value, "merged_features_binary", "features_sparse", "features.gz"
        )
        for key, value in data_paths.items()
    }

    result_paths = {
        key: os.path.join(value, "cohort", args.result_file_name)
        for key, value in data_paths.items()
    }

    # num_folds_dict = {"starr": 5, "optum_dod": 5}
    num_folds = 5

    cohort = pd.read_parquet(cohort_paths[args.db_key])
    features = joblib.load(features_paths[args.db_key])

    config_dict = {
        "label_col": "days_until_event",
        "event_indicator_var_name": "event_indicator",
        "num_bins": 20,
        "row_id_col": "row_id",
        "fold_id_test": ["eval", "test"],
        "disable_metric_logging": True,
        "num_hidden": args.num_hidden,
        "hidden_dim": 128,
        "lr": 1e-4,
        "num_epochs": 1000,
        "batch_size": 512,
        "early_stopping": True,
        "early_stopping_patience": 10,
        "input_dim": features.shape[1],
        "sparse_mode": "list",
    }

    result = {}
    for val_fold_id in [str(i + 1) for i in range(num_folds)]:
        print(f"Cohort: {args.db_key}, fold_id: {val_fold_id}")
        result[val_fold_id] = fit_and_eval_discrete_time_model(
            features=features,
            cohort=cohort,
            time_horizon=10 * 365.25,
            censoring_model=True,
            fold_id=val_fold_id,
            **config_dict,
        )
    result_df = (
        pd.concat(result)
        .reset_index(level=-1, drop=True)
        .rename_axis("val_fold_id")
        .reset_index()
        .drop(columns=["phase"])
    )
    cohort_with_result = cohort.merge(result_df)
    result_df_aggregated = (
        cohort_with_result[["prediction_id", "val_fold_id", "pred_probs"]]
        .groupby("prediction_id")
        .agg(censoring_survival_function=("pred_probs", "mean"))
        .reset_index()
        .assign(ipcw_weight=lambda x: 1 / x.censoring_survival_function)
        .sort_values("ipcw_weight")
    )
    cohort_with_result = cohort.merge(result_df_aggregated)
    cohort_with_result.to_parquet(result_paths[args.db_key], index=False)
