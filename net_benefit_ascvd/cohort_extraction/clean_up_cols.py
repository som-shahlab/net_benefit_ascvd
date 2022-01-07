import os
import pandas as pd
import configargparse as argparse

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

if __name__ == "__main__":
    args = parser.parse_args()
    cohort_df = pd.read_parquet(args.source_cohort_path)
    cohort_df = cohort_df.assign(
        has_diabetes_type2_history=lambda x: x["has_diabetes_type2_history"]
        .where(lambda y: y != 1, "diabetes_type2_present")
        .where(lambda y: y != 0, "diabetes_type2_absent"),
        has_diabetes_type1_history=lambda x: x["has_diabetes_type1_history"]
        .where(lambda y: y != 1, "diabetes_type1_present")
        .where(lambda y: y != 0, "diabetes_type1_absent"),
        has_ckd_history=lambda x: x["has_ckd_history"]
        .where(lambda y: y != 1, "ckd_present")
        .where(lambda y: y != 0, "ckd_absent"),
        has_ra_history=lambda x: x["has_ra_history"]
        .where(lambda y: y != 1, "ra_present")
        .where(lambda y: y != 0, "ra_absent"),
    )
    cohort_df.to_parquet(args.target_cohort_path, index=False, engine="pyarrow")
