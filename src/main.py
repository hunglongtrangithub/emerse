import json
import os
from pathlib import Path

import boto3
from moto import mock_aws
import torch
from flask import Flask, jsonify, request
from healthcheck import EnvironmentDump, HealthCheck

from .models import ModelRegistry
from .process_report import get_reports_with_table
from .config import logger, setup_testing_s3, MODE, DEBUG, REPORT_TEXT_COLUMN

app = Flask(__name__)


health = HealthCheck()
envdump = EnvironmentDump()


def models_healthy():
    try:
        if not model_registry.models_loaded:
            return False, "Models are not loaded"
        is_healthy, messages = model_registry.check_all_models_health()
        if not is_healthy:
            return False, "\n".join(messages)
        return True, "All models are healthy"
    except Exception as e:
        return False, str(e)


def cuda_available():
    if torch.cuda.is_available():
        return True, "CUDA is available"
    else:
        return False, "CUDA is not available"


def s3_accessible():
    try:
        s3 = boto3.resource("s3")
        bucket_name = os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            return False, "S3_BUCKET_NAME environment variable is not set"
        bucket = s3.Bucket(bucket_name)
        # Try to list objects to check if the bucket is accessible
        _ = list(bucket.objects.limit(1))
        return True, f"Access to S3 bucket {bucket_name} verified"
    except Exception as e:
        return False, str(e)


health.add_check(models_healthy)
health.add_check(cuda_available)
health.add_check(s3_accessible)


def application_data():
    return {
        "description": "NLP Application with Flask",
    }


envdump.add_section("application", application_data)

app.add_url_rule("/health_check", "healthcheck", view_func=lambda: health.run())
app.add_url_rule("/environment", "environment", view_func=lambda: envdump.run())


@app.after_request
def after_request(response):
    logger.info(
        "{} {} {} {} {}",
        request.remote_addr,
        request.method,
        request.scheme,
        request.full_path,
        response.status,
    )
    return response


@app.before_request
def log_request_info():
    logger.debug(f"Received {request.method} request on {request.path}")
    logger.debug(f"Headers: {request.headers}")
    logger.debug(f"Body: {request.get_data()}")


@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(str(e), exc_info=True)
    return str(e), 500


@app.route("/", methods=["GET"])
def hello_world():
    return jsonify({"body": "Hello World"}), 200


def get_json_objects(file_content: str) -> list[dict[str, str]]:
    # Split the file content into lines
    lines = file_content.splitlines()

    # Initialize a list to hold parsed JSON objects
    json_objects = []

    for i, line in enumerate(lines):
        # Strip trailing commas and whitespace from each line
        line = line.rstrip(",").strip()

        # Parse each line as JSON and append it to the list
        if line:  # Make sure the line is not empty
            try:
                json_obj = json.loads(line)
                json_objects.append(json_obj)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from line {i + 1}: {e}")
                continue
    if not json_objects:
        raise json.JSONDecodeError("No valid JSON objects found", file_content, 0)
    return json_objects


def to_json_lines(reports: list[dict[str, str]]) -> str:
    json_lines = "\n".join(
        json.dumps(report, separators=(",", ":")) for report in reports
    )
    return json_lines


def process_report_files(file_loader, file_saver, test=False):
    count = 0
    for file_id, json_input in file_loader():
        logger.info(f"Processing file: {file_id}")

        if not isinstance(json_input, list):
            logger.error(f"Invalid JSON input in {file_id}. Must be a list of reports.")
            continue

        logger.info(f"Number of new reports in {file_id}: {len(json_input)}")

        try:
            if test:
                json_input = json_input[:5]
            reports = get_reports_with_table(json_input, model_registry)
        except Exception as e:
            logger.error(f"Error in processing reports from {file_id}: {e}")
            continue

        count += len(reports)
        logger.info(
            f"{file_id}: Done predicting new reports. Number of processed reports so far: {count}"
        )

        file_saver(file_id, reports)
    return count


def load_files_from_directory(input_directory: str):
    input_dir = Path(input_directory)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' not found.")

    filepaths = input_dir.glob("*.jsonl")
    for filepath in filepaths:
        logger.info(f"Reading file: {filepath}")
        try:
            with open(filepath, "r") as file:
                file_content = file.read()
                yield filepath.name, get_json_objects(file_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {filepath}: {e}")
            continue


def save_files_to_directory(
    output_directory: str, filepath: str, reports: list[dict[str, str]]
):
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)

    fileout = output_dir / filepath
    logger.info(f"Writing JSON to file: {fileout}")
    with fileout.open("w") as file:
        file.write(to_json_lines(reports))

    html_out = output_dir / (fileout.stem + ".html")
    for report in reports:
        if REPORT_TEXT_COLUMN in report:
            logger.info(f"Writing HTML to file: {html_out}")
            with html_out.open("w") as file:
                file.write(report[REPORT_TEXT_COLUMN])
            break


def load_files_from_s3(bucket_name: str, prefix: str = ""):
    s3 = boto3.resource("s3")
    logger.info(f"Reading files. Bucket: {bucket_name} | Prefix: {prefix}")
    bucket = s3.Bucket(bucket_name)
    objects_list = bucket.objects.filter(Prefix=prefix)

    for obj in objects_list:
        obj_key = obj.key
        if obj_key.endswith("/"):
            logger.info(f"Skipping directory: {obj_key}")
            continue
        logger.info(f"Reading object: {obj_key}")
        obj_body = obj.get()["Body"].read().decode("utf-8")
        try:
            yield obj_key, get_json_objects(obj_body)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {obj_key}: {e}")
            continue


def save_files_to_s3(
    bucket_name: str, output_prefix: str, obj_key: str, reports: list[dict[str, str]]
):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    updated_json = to_json_lines(reports).encode("utf-8")
    new_key = output_prefix + obj_key.split("/")[-1]
    logger.info(f"Saving JSON to S3 with key: {new_key}")
    bucket.put_object(
        Body=updated_json,
        Key=new_key,
    )


@app.route("/predict_test", methods=["GET"])
def predict_test():
    try:
        count = process_report_files(
            file_loader=lambda: load_files_from_directory("./input"),
            file_saver=lambda filepath, reports: save_files_to_directory(
                "./output", filepath, reports
            ),
            test=True,
        )
        return jsonify({"processed reports count": count}), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/predict", methods=["GET"])
def process_files():
    s3_bucket_name = os.getenv("S3_BUCKET_NAME")
    if not s3_bucket_name:
        return jsonify({"error": "S3_BUCKET_NAME environment variable is not set"}), 500
    input_prefix = request.args.get("input_prefix", "")
    output_prefix = request.args.get("output_prefix", "output/")
    try:
        count = process_report_files(
            file_loader=lambda: load_files_from_s3(s3_bucket_name, input_prefix),
            file_saver=lambda obj_key, reports: save_files_to_s3(
                s3_bucket_name, output_prefix, obj_key, reports
            ),
        )
        return jsonify({"processed reports count": count}), 200
    except Exception as e:
        logger.error(f"Error in processing S3 files: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/list_s3_bucket", methods=["GET"])
def list_s3_bucket():
    bucket_name = os.getenv("S3_BUCKET_NAME")
    max_count = int(request.args.get("max_count", 10))
    if not bucket_name:
        return jsonify({"error": "S3_BUCKET_NAME environment variable is not set"}), 500
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    objects = list(bucket.objects.limit(max_count))
    return (
        jsonify(
            {
                "contents": [
                    {obj.key: obj.get()["Body"].read().decode("utf-8")}
                    for obj in objects
                ]
            }
        ),
        200,
    )


def main():
    global model_registry
    model_registry = ModelRegistry("./models", "./saved_models")

    logger.info(f"Mode: {MODE}. Debug: {DEBUG}")
    logger.info("Starting NLP application...")
    try:
        logger.info("Loading models...")
        model_registry.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error in loading models: {e}")
        return

    def run_app():
        if MODE == "testing":
            with mock_aws():
                setup_testing_s3()
                from waitress import serve

                serve(app, host="0.0.0.0", port=os.getenv("PORT", 5000))
        else:
            from waitress import serve

            serve(app, host="0.0.0.0", port=os.getenv("PORT", 5000))

    run_app()
