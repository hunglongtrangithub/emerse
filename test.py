import os
import json

import pytest
import boto3
from moto import mock_aws

from src.process_report import get_reports_with_table
from src.models import load_models
from src.main import s3_accessible, load_files_from_s3, save_files_to_s3


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["S3_BUCKET_NAME"] = "test-bucket"


@pytest.fixture(scope="function")
def mocked_aws(aws_credentials):
    """
    Mock all AWS interactions
    Requires you to create your own boto3 clients
    """
    with mock_aws():
        yield


def test_batch():
    load_models()
    reports = json.loads(open("input/test_batch.json").read())
    reports = get_reports_with_table(reports)
    json.dump(reports, open("output/test_batch.json", "w"), indent=2)


def test_s3_accessible(mocked_aws):
    bucket_name = "test-bucket"
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=bucket_name)

    assert s3_accessible()[0] == True


def test_load_files_from_s3(mocked_aws):
    # Set up the mocked S3 environment
    bucket_name = "test-bucket"
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=bucket_name)

    # Add a test file to the bucket
    test_data = {"key": "value"}
    conn.Bucket(bucket_name).put_object(
        Key="test_file.json", Body=json.dumps(test_data)
    )

    # Test the load_files_from_s3 function
    result = list(load_files_from_s3(bucket_name))

    assert len(result) == 1  # Ensure one file is returned
    assert result[0][0] == "test_file.json"  # Ensure the key is correct
    assert result[0][1] == test_data  # Ensure the data is correctly loaded


def test_load_files_from_s3_invalid_json(mocked_aws):
    # Set up the mocked S3 environment
    bucket_name = "test-bucket"
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=bucket_name)

    # Add a test file with invalid JSON to the bucket
    conn.Bucket(bucket_name).put_object(
        Key="invalid_test_file.json", Body="This is not JSON"
    )

    # Test the load_files_from_s3 function
    result = list(load_files_from_s3(bucket_name))

    assert len(result) == 0  # Ensure no valid JSON files are returned


def test_save_files_to_s3(mocked_aws):
    # Set up the mocked S3 environment
    bucket_name = "test-bucket"
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=bucket_name)

    # Prepare test data to save
    test_data = {"key": "value"}
    obj_key = "saved_test_file.json"

    # Test the save_files_to_s3 function
    save_files_to_s3(bucket_name, obj_key, test_data)

    # Verify the data was saved correctly
    s3_object = conn.Object(bucket_name, obj_key).get()
    saved_data = json.loads(s3_object["Body"].read().decode("utf-8"))

    assert saved_data == test_data  # Ensure the saved data matches the input


if __name__ == "__main__":
    # test_batch()
    print("All tests passed")
