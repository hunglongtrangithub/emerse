import os
from pathlib import Path
import sys

from loguru import logger
from dotenv import load_dotenv
import boto3

load_dotenv()


# Environment variables
DEBUG = os.getenv("DEBUG", False)
MODE = os.getenv("MODE", "development")
# Maximum length of input sequence to truncate
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 4096))
# Batch size for prediction
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
# JSON key for the report text
REPORT_TEXT_COLUMN = os.getenv("REPORT_TEXT_COLUMN", "RPT_TEXT")

# Configure Loguru based on MODE
if MODE == "production":
    log_file = Path("./log/nlp_app.log")
    log_file.parent.mkdir(exist_ok=True)
    logger.add(log_file, level="INFO", rotation="10 MB")
else:
    logger.remove()  # Remove default stderr logging
    logger.add(sys.stderr, level="DEBUG" if DEBUG else "INFO")


# Set up testing environment for AWS S3
def setup_testing_s3():
    bucket_name = "test-bucket"
    # Set up fake AWS credentials
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["S3_BUCKET_NAME"] = bucket_name

    # Mock S3 and create a test bucket with objects
    def setup_mock_s3():
        s3 = boto3.resource("s3")
        s3.create_bucket(Bucket=bucket_name)

        # Add some test objects
        s3.Bucket(bucket_name).put_object(
            Key="test_file_1.jsonl", Body='{"key": "value1"}\n{"key": "value1"}'
        )
        s3.Bucket(bucket_name).put_object(
            Key="test_file_2.jsonl", Body='{"key": "value2"}\n{"key": "value2"}'
        )
        test_file = "test_batch.jsonl"
        file_content = open(f"./input/{test_file}").read()
        s3.Bucket(bucket_name).put_object(Key="test_batch.jsonl", Body=file_content)
        logger.info(f"Mock S3 bucket '{bucket_name}' created with test objects.")

    # Execute the mock setup
    setup_mock_s3()
