from time import strftime
from pathlib import Path
import json
import torch
import boto3
from flask import Flask, jsonify, request
from healthcheck import HealthCheck, EnvironmentDump
from predict import get_reports_with_table
from config import logger, S3_BUCKET_NAME

app = Flask(__name__)


health = HealthCheck()
envdump = EnvironmentDump()


def models_loaded():
    try:
        for model_variable in [
            "tokenizer_pat",
            "model_pat",
            "tokenizer_lat",
            "model_lat",
            "tokenizer_gra",
            "model_gra",
            "tokenizer_lym",
            "model_lym",
        ]:
            if model_variable not in globals():
                return False, f"Model {model_variable} is not loaded"
            return True, "All models loaded successfully"
        else:
            return False, "One or more models are not loaded"
    except Exception as e:
        return False, str(e)


def cuda_available():
    if torch.cuda.is_available():
        return True, "CUDA is available"
    else:
        return False, "CUDA is not available"


def s3_accessible():
    try:
        if not S3_BUCKET_NAME:
            return False, "S3_BUCKET_NAME environment variable is not set"
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(S3_BUCKET_NAME)
        # Try to list objects to check if the bucket is accessible
        _ = list(bucket.objects.limit(1))
        return True, f"Access to S3 bucket {S3_BUCKET_NAME} verified"
    except Exception as e:
        return False, str(e)


health.add_check(models_loaded)
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
    timestamp = strftime("[%Y-%b-%d %H:%M]")
    logger.info(
        "%s %s %s %s %s %s",
        timestamp,
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
    logger.error(f"An error occurred: {str(e)}", exc_info=True)
    return jsonify(error=str(e)), 500


@app.route("/", methods=["GET"])
def hello_world():
    return jsonify({"body": "Hello World"}), 200


def process_report_files(file_loader, file_saver):
    count = 0
    for file_id, json_input in file_loader():
        logger.info(f"Processing file: {file_id}")
        
        if not isinstance(json_input, list):
            logger.error(f"Invalid JSON input in {file_id}. Must be a list of reports.")
            continue
        
        logger.info(f"Number of new reports in {file_id}: {len(json_input)}")

        try:
            reports = get_reports_with_table(json_input)
        except Exception as e:
            logger.error(f"Error in processing reports from {file_id}: {e}")
            continue

        count += len(reports)
        logger.info(f"{file_id}: Done predicting new reports. Number of processed reports: {count}")

        file_saver(file_id, reports)

    return {
        "statusCode": 200,
        "body": json.dumps({"processed reports count": count}, indent=4),
    }


def load_files_from_directory(input_directory):
    input_directory = Path(input_directory)
    if not input_directory.exists():
        raise FileNotFoundError(f"Input directory '{input_directory}' not found.")

    filepaths = input_directory.glob("*.json")
    for filepath in filepaths:
        with open(filepath, "r") as file:
            yield filepath.name, json.load(file)


def save_files_to_directory(output_directory, filepath, reports):
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True)

    fileout = output_directory / filepath
    logger.info(f"Writing JSON to file: {fileout}")
    with fileout.open("w") as file:
        json.dump(reports, file, indent=4)
    
    html_out = output_directory / (fileout.stem + ".html")
    for report in reports:
        if "RPT_TEXT" in report:
            logger.info(f"Writing HTML to file: {html_out}")
            with html_out.open("w") as file:
                file.write(report["RPT_TEXT"])
            break


def load_files_from_s3():
    s3 = boto3.resource("s3")
    logger.info(f"Reading files from bucket: {S3_BUCKET_NAME}")
    bucket = s3.Bucket(S3_BUCKET_NAME)
    objects_list = bucket.objects.all()

    for obj in objects_list:
        obj_key = obj.key
        obj_body = obj.get()["Body"].read().decode("utf-8")
        yield obj_key, json.loads(obj_body)


def save_files_to_s3(bucket_name, obj_key, reports):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    updated_json = json.dumps(reports, indent=4).encode("utf-8")
    logger.info(f"Saving JSON to S3 with key: {obj_key}")
    bucket.put_object(
        Body=updated_json,
        Key=obj_key,
    )


@app.route("/predict_test", methods=["GET"])
def predict_test():
    try:
        response = process_report_files(
            file_loader=lambda: load_files_from_directory("./input"),
            file_saver=lambda filepath, reports: save_files_to_directory("./output", filepath, reports)
        )
        return response
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/predict", methods=["GET"])
def process_files():
    try:
        response = process_report_files(
            file_loader=load_files_from_s3,
            file_saver=lambda obj_key, reports: save_files_to_s3(S3_BUCKET_NAME, obj_key, reports)
        )
        return response
    except Exception as e:
        logger.error(f"Error in processing S3 files: {e}")
        return jsonify({"error": str(e)}), 500
