import os

# Set dummy AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
import json

import boto3
from moto import mock_aws

from process_report import get_reports_with_table
from models import load_models
from main import s3_accessible, load_files_from_s3, save_files_to_s3


def test_batch():
    load_models()
    reports = json.loads(open("input/test_batch.json").read())
    reports = get_reports_with_table(reports)
    json.dump(reports, open("output/test_batch.json", "w"), indent=2)


@mock_aws
def test_s3_accessible():
    bucket_name = "test-bucket"
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=bucket_name)

    assert s3_accessible(bucket_name) == True


if __name__ == "__main__":
    test_batch()
