import os
import json

import pytest
import boto3
from moto import mock_aws

from src.models import model_registry
from src.process_report import get_reports_with_table
from src.main import (
    s3_accessible,
    load_files_from_s3,
    save_files_to_s3,
    to_json_lines,
    get_json_objects,
)


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
    model_registry.load_models()
    reports = get_json_objects(open("input/test_batch.jsonl").read())
    reports = get_reports_with_table(reports)
    with open("output/test_batch.jsonl", "w") as f:
        f.write(to_json_lines(reports))


def test_s3_accessible(mocked_aws):
    bucket_name = "test-bucket"
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=bucket_name)

    assert s3_accessible()[0]


def test_load_files_from_s3(mocked_aws):
    # Set up the mocked S3 environment
    bucket_name = "test-bucket"
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=bucket_name)

    # Add a test file to the bucket
    obj_key = "test_file.jsonl"
    test_data = [{"key": "value"}, {"key": "value2"}]
    obj_body = to_json_lines(test_data)
    conn.Bucket(bucket_name).put_object(Key=obj_key, Body=obj_body.encode("utf-8"))

    # Test the load_files_from_s3 function
    result = list(load_files_from_s3(bucket_name))

    assert len(result) == 1  # Ensure one file is returned
    assert result[0][0] == obj_key  # Ensure the key is correct
    assert result[0][1] == test_data  # Ensure the data is correctly loaded


def test_load_files_from_s3_invalid_json(mocked_aws):
    # Set up the mocked S3 environment
    bucket_name = "test-bucket"
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=bucket_name)

    # Add a test file with invalid JSON to the bucket
    conn.Bucket(bucket_name).put_object(
        Key="invalid_test_file.jsonl", Body="This is not JSON".encode("utf-8")
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
    test_data = [{"key": "value"}, {"key": "value2"}]
    obj_key = "input/saved_test_file.jsonl"
    output_prefix = "output/"
    # Test the save_files_to_s3 function
    save_files_to_s3(bucket_name, output_prefix, obj_key, test_data)

    # Verify the data was saved correctly
    new_key = output_prefix + obj_key.split("/")[-1]
    s3_object = conn.Object(bucket_name, new_key).get()
    saved_data = get_json_objects(s3_object["Body"].read().decode("utf-8"))

    assert saved_data == test_data  # Ensure the saved data matches the input


if __name__ == "__main__":
    # test_batch()
    print("All tests passed")
