import configargparse as argparse
import os
from net_benefit_ascvd.prediction_utils.extraction_utils.database import BQDatabase


parser = argparse.ArgumentParser()

parser.add_argument("--rs_dataset", type=str)
parser.add_argument("--gcloud_project", type=str)
parser.add_argument("--rs_dataset_project", type=str)
parser.add_argument("--cohort_name", type=str, default="ascvd_10yr")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument(
    "--google_application_credentials",
    type=str,
    default=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
)
parser.add_argument("--cohort_filename", default="cohort")

if __name__ == "__main__":

    args = parser.parse_args()
    config_dict = args.__dict__

    db = BQDatabase(**config_dict)

    query = """
        SELECT * 
        FROM {rs_dataset_project}.{rs_dataset}.{cohort_name}
    """.format(
        **config_dict
    )

    cohort_df = db.read_sql_query(query, use_bqstorage_api=True)

    cohort_path = os.path.join(args.data_path, "cohort")
    os.makedirs(cohort_path, exist_ok=True)
    cohort_df.to_parquet(
        os.path.join(cohort_path, "{}.parquet".format(args.cohort_filename)),
        engine="pyarrow",
        index=False,
    )
