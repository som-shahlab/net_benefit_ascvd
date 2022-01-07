import configargparse as argparse
import os

from net_benefit_ascvd.cohort_extraction.cohort import (
    ASCVDCohort,
    ASCVDDemographicsCohort,
    get_transform_query_sampled,
    get_label_config,
)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str)
parser.add_argument("--rs_dataset", type=str)
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--gcloud_project", type=str)
parser.add_argument("--dataset_project", type=str)
parser.add_argument("--rs_dataset_project", type=str)
parser.add_argument("--temp_dataset_project", type=str)
parser.add_argument("--temp_dataset", type=str, default="temp_dataset")
parser.add_argument("--cohort_name", type=str, default="ascvd_10yr")
parser.add_argument("--cohort_name_sampled", type=str, default="ascvd_10yr_sampled")
parser.add_argument(
    "--sample", dest="sample", action="store_true", help="Whether to sample the data"
)
parser.add_argument("--horizon", type=str, default="10yr")
parser.add_argument("--db_key", type=str, default="starr")

parser.add_argument(
    "--has_birth_datetime", dest="has_birth_datetime", action="store_true"
)

parser.add_argument(
    "--overwrite_concept_tables", dest="overwrite_concept_tables", action="store_true"
)

parser.add_argument(
    "--google_application_credentials",
    type=str,
    default=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
)

parser.set_defaults(
    has_birth_datetime=False, overwrite_concept_tables=False, sample=False
)

if __name__ == "__main__":
    args = parser.parse_args()

    label_config = get_label_config(args.db_key)

    config_dict = {**args.__dict__, **label_config[args.horizon]}

    cohort = ASCVDCohort(**config_dict)
    cohort.create_concept_tables(overwrite=args.overwrite_concept_tables)
    destination_table = "{rs_dataset_project}.{rs_dataset}.{cohort_name}".format_map(
        config_dict
    )
    print(destination_table)
    cohort.db.execute_sql_to_destination_table(
        cohort.get_transform_query(), destination=destination_table
    )
    cohort_demographics = ASCVDDemographicsCohort(**config_dict)
    cohort_demographics.db.execute_sql_to_destination_table(
        cohort_demographics.get_transform_query(), destination=destination_table
    )

    where_clause = "WHERE has_cvd_history = 0 AND has_statin_history = 0"
    cohort.db.execute_sql_to_destination_table(
        get_transform_query_sampled(
            source_table=destination_table,
            sample_where_clause=where_clause,
        ),
        destination=destination_table,
    )
    if args.sample:
        where_clause = f"{where_clause} AND sampled = 1"

    sample_query = f"""
        SELECT * 
        FROM {destination_table}
        {where_clause}
    """
    cohort.db.execute_sql_to_destination_table(
        sample_query,
        destination="{rs_dataset_project}.{rs_dataset}.{cohort_name_sampled}".format_map(
            config_dict
        ),
    )
