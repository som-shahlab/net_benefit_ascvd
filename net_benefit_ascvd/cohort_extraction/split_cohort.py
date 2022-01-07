import pandas as pd
import argparse
from net_benefit_ascvd.cohort_extraction.util import patient_split_cv

parser = argparse.ArgumentParser()

parser.add_argument(
    "--source_cohort_path",
    type=str,
    help="The source file",
    required=True,
)

parser.add_argument(
    "--target_cohort_path",
    type=str,
    help="The destination file",
    required=True,
)

parser.add_argument(
    "--nfold",
    type=int,
    default=5,
)

parser.add_argument(
    "--test_frac",
    type=float,
    default=0.25,
)

parser.add_argument(
    "--seed",
    type=int,
    default=9526,
)

parser.add_argument(
    "--strata_vars",
    type=str,
    nargs="*",
    required=False,
    default=None,
    help="Variables to stratify the sampling on",
)

parser.add_argument(
    "--features_row_id_map_path",
    type=str,
    default=None,
    help="Maps identifiers in the cohort dataframe to rows of the feature matrix",
)


if __name__ == "__main__":
    args = parser.parse_args()

    cohort = pd.read_parquet(args.source_cohort_path)
    if "fold_id" in cohort.columns:
        cohort = cohort.drop(columns="fold_id")

    if args.features_row_id_map_path is not None:
        features_row_id_map = pd.read_parquet(args.features_row_id_map_path)
        cohort = cohort.merge(features_row_id_map).rename(
            columns={"features_row_id": "row_id"}
        )

    eval_frac = float((1 - (args.test_frac)) / (args.nfold + 1))
    if args.strata_vars is not None:
        split_cohort = (
            cohort.groupby(args.strata_vars)
            .apply(
                lambda x: patient_split_cv(
                    x,
                    test_frac=args.test_frac,
                    eval_frac=args.eval_frac,
                    nfold=args.nfold,
                    seed=args.seed,
                )
            )
            .reset_index(drop=True)
        )
    else:
        split_cohort = patient_split_cv(
            cohort,
            test_frac=args.test_frac,
            eval_frac=eval_frac,
            nfold=args.nfold,
            seed=args.seed,
        )

    split_cohort.to_parquet(args.target_cohort_path, index=False)
