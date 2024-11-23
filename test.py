import os

import pytest
import boto3
from moto import mock_aws

from src.models import ModelRegistry, PathologyPrediction, MobilityPrediction
from src.process_report import get_reports_with_table, generate_html_report
from src.main import (
    s3_accessible,
    load_files_from_s3,
    save_files_to_s3,
    to_json_lines,
    get_json_objects,
)


def test_generate_html_report():
    # Mock data for pathology predictions
    batch_predictions = [
        PathologyPrediction(
            field="Pathology",
            field_name="PAT",
            value=["Malignant", "Benign"],
            max_prob=[0.98, 0.95],
        ),
        PathologyPrediction(
            field="Laterality",
            field_name="LAT",
            value=["Left", "Right"],
            max_prob=[0.92, 0.88],
        ),
        PathologyPrediction(
            field="Grade",
            field_name="GRA",
            value=["Grade 2", "Grade 3"],
            max_prob=[0.96, 0.91],
        ),
        PathologyPrediction(
            field="Lymph Node Metastasis",
            field_name="LYM",
            value=["Positive", "Negative"],
            max_prob=[0.85, 0.89],
        ),
    ]

    # Mock data for mobility predictions
    batch_entity_indexes = {
        "Action": [
            MobilityPrediction(
                tokenized_input=["Patient", "was", "walking", "slowly", "."],
                output_tags=["O", "O", "B-Action", "I-Action", "O"],
                entity_indexes=[
                    [16, 31]
                ],  # Start and end positions for "walking slowly"
                extracted_entities=["walking slowly"],
            ),
            MobilityPrediction(
                tokenized_input=["The", "subject", "ran", "quickly", "."],
                output_tags=["O", "O", "B-Action", "I-Action", "O"],
                entity_indexes=[[10, 23]],  # Start and end positions for "ran quickly"
                extracted_entities=["ran quickly"],
            ),
        ],
        "Assistant": [
            MobilityPrediction(
                tokenized_input=["The", "nurse", "helped", "the", "patient", "."],
                output_tags=["O", "B-Assistant", "I-Assistant", "O", "O", "O"],
                entity_indexes=[[4, 17]],  # Start and end positions for "nurse helped"
                extracted_entities=["nurse helped"],
            ),
            MobilityPrediction(
                tokenized_input=["Doctor", "provided", "assistance", "."],
                output_tags=["B-Assistant", "I-Assistant", "I-Assistant", "O"],
                entity_indexes=[
                    [0, 25]
                ],  # Start and end positions for "Doctor provided assistance"
                extracted_entities=["Doctor provided assistance"],
            ),
        ],
        "Mobility": [
            MobilityPrediction(
                tokenized_input=["Patient", "can", "move", "independently", "."],
                output_tags=["O", "O", "B-Mobility", "I-Mobility", "O"],
                entity_indexes=[
                    [11, 33]
                ],  # Start and end positions for "move independently"
                extracted_entities=["move independently"],
            ),
            MobilityPrediction(
                tokenized_input=["Subject", "needs", "assistance", "to", "stand", "."],
                output_tags=["O", "O", "B-Mobility", "O", "I-Mobility", "O"],
                entity_indexes=[
                    [12, 28]
                ],  # Start and end positions for "assistance to stand"
                extracted_entities=["assistance to stand"],
            ),
        ],
        "Quantification": [
            MobilityPrediction(
                tokenized_input=[
                    "The",
                    "subject",
                    "walked",
                    "for",
                    "10",
                    "meters",
                    ".",
                ],
                output_tags=[
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-Quantification",
                    "I-Quantification",
                    "O",
                ],
                entity_indexes=[[27, 37]],  # Start and end positions for "10 meters"
                extracted_entities=["10 meters"],
            ),
            MobilityPrediction(
                tokenized_input=["Patient", "ran", "for", "5", "minutes", "."],
                output_tags=[
                    "O",
                    "O",
                    "O",
                    "B-Quantification",
                    "I-Quantification",
                    "O",
                ],
                entity_indexes=[[19, 30]],  # Start and end positions for "5 minutes"
                extracted_entities=["5 minutes"],
            ),
        ],
    }

    # Mock report text
    report_text = """
    The pathology report indicates a malignant tumor on the left side with Grade 2 classification.
    Additionally, there is evidence of lymph node metastasis. The patient walked slowly but needs assistance.
    """

    # Mock index
    index = 0  # Generate the HTML report
    html_content = generate_html_report(
        report_text, index, batch_predictions, batch_entity_indexes
    )

    with open("output/test_generate_html_report.html", "w") as f:
        f.write(html_content)

    print("HTML report generated successfully")


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
    model_registry = ModelRegistry("./models", "./saved_models")
    model_registry.load_models()
    reports = get_json_objects(open("input/test_batch.jsonl").read())
    reports = get_reports_with_table(reports, model_registry)
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
