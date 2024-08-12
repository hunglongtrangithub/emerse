import os
from pathlib import Path
import json
import logging
from time import strftime

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from healthcheck import HealthCheck, EnvironmentDump
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import boto3
from bs4 import BeautifulSoup
from yattag import Doc

DEBUG = os.getenv("DEBUG") or False
MAX_LENGTH = 4096  # Maximum length of input sequence to truncate

if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    Path("./log").mkdir(exist_ok=True)
    logging.basicConfig(filename="./log/nlp_app.log", level=logging.INFO)

logger = logging.getLogger(__name__)
logger.info(f"Debug mode: {DEBUG}")
# Todo: Discuss with IT on a protocol to find out which files have been processed,
# and contain the results, which have failed and probably why, and intergrate it into the code.


def load_models():
    global device
    global tokenizer_pat, model_pat
    global tokenizer_lat, model_lat
    global tokenizer_gra, model_gra
    global tokenizer_lym, model_lym

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading models on device: {device}")

    model_name_pat = "./models/pat_BigBird_mimic_3_path_3_epoch30_new"
    tokenizer_pat = AutoTokenizer.from_pretrained(model_name_pat)
    model_pat = AutoModelForSequenceClassification.from_pretrained(
        model_name_pat, num_labels=2, ignore_mismatched_sizes=True
    )
    model_pat = model_pat.to(device)
    logger.info(f"Pathology prediction model loaded successfully: {model_name_pat}")

    model_name_lat = "./models/lat_BigBird_mimic_3_path_3_epoch30"
    tokenizer_lat = AutoTokenizer.from_pretrained(model_name_lat)
    model_lat = AutoModelForSequenceClassification.from_pretrained(
        model_name_lat, num_labels=5
    )
    model_lat = model_lat.to(device)
    logger.info(f"Laterality prediction model loaded successfully: {model_name_lat}")

    model_name_gra = "./models/gra_BigBird_mimic_3_path_3_epoch30"
    tokenizer_gra = AutoTokenizer.from_pretrained(model_name_gra)
    model_gra = AutoModelForSequenceClassification.from_pretrained(
        model_name_gra, num_labels=5
    )
    model_gra = model_gra.to(device)
    logger.info(f"Grade prediction model loaded successfully: {model_name_gra}")

    model_name_lym = "./models/lym_BigBird_mimic_3_path_3_epoch30"
    tokenizer_lym = AutoTokenizer.from_pretrained(model_name_lym)
    model_lym = AutoModelForSequenceClassification.from_pretrained(
        model_name_lym, num_labels=4, ignore_mismatched_sizes=True
    )
    model_lym = model_lym.to(device)
    logger.info(
        f"Lymph node metastasis prediction model loaded successfully: {model_name_lym}"
    )
    return

# TODO: think about doing batch predictions for multiple reports.
def model_predict(
    input_text, model, tokenizer, device, tokenizer_kwargs={}, model_kwargs={}
):
    # Tokenize input sequence with truncation
    inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        **tokenizer_kwargs,
    ).to(device)

    model.to(device)
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs, **model_kwargs)
        logits = outputs.logits

    # Apply softmax to get confidence scores
    probs = torch.nn.functional.softmax(logits, dim=1)
    max_prob1, predicted_label_id = torch.max(probs, dim=1)

    predicted_label1 = model.config.id2label[predicted_label_id.item()]

    return predicted_label1, max_prob1.item()

def predict(report):
    predicted_label_pat, max_prob_pat = model_predict(
        report, model_pat, tokenizer_pat, device
    )
    predicted_label_lat, max_prob_lat = model_predict(
        report, model_lat, tokenizer_lat, device
    )
    predicted_label_gra, max_prob_gra = model_predict(
        report, model_gra, tokenizer_gra, device
    )
    predicted_label_lym, max_prob_lym = model_predict(
        report, model_lym, tokenizer_lym, device
    )

    return [
        {
            "field": "PAT",
            "field_name": "Pathology",
            "value": predicted_label_pat,
            "max_prob": max_prob_pat,
        },
        {
            "field": "LAT",
            "field_name": "Laterality",
            "value": predicted_label_lat,
            "max_prob": max_prob_lat,
        },
        {
            "field": "GRA",
            "field_name": "Grade",
            "value": predicted_label_gra,
            "max_prob": max_prob_gra,
        },
        {
            "field": "LYM",
            "field_name": "Lymph Node Metastasis",
            "value": predicted_label_lym,
            "max_prob": max_prob_lym,
        },
    ]


def get_report_with_table(report_html: str) -> str:
    """
    Extracts the report text from the HTML and adds a table with the predictions.
    Assumes report is in proper HTML format (properly closed tags, etc.).
    """
    soup = BeautifulSoup(report_html, "html.parser")
    if soup.body is None:
        raise ValueError("No body tag found in the report.")

    report = soup.body.get_text(strip=True)
    if not report:
        raise ValueError("No text found in the report.")

    predictions = predict(report)

    doc, tag, text = Doc().tagtext()
    with tag("table"):
        with tag("tr"):
            with tag("th"):
                text("")
            for prediction in predictions:
                with tag("th"):
                    text(prediction["field_name"])
        with tag("tr"):
            with tag("th"):
                text("Value")
            for prediction in predictions:
                with tag("td"):
                    with tag("field", name=prediction["field"], type="string"):
                        text(prediction["value"])
        with tag("tr"):
            with tag("th"):
                text("Max Prob")
            for prediction in predictions:
                with tag("td"):
                    with tag(
                        "field", name=f"{prediction['field']}_prob", type="string"
                    ):
                        text(prediction["max_prob"])

    table_html = doc.getvalue()
    soup.body.append(BeautifulSoup(table_html, "html.parser"))
    return str(soup)


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


health.add_check(models_loaded)
health.add_check(cuda_available)
health.add_check(s3_accessible)


def application_data():
    return {
        "description": "NLP Application with Flask",
    }


envdump.add_section("application", application_data)

app.add_url_rule("/healthcheck", "healthcheck", view_func=lambda: health.run())
app.add_url_rule("/environment", "environment", view_func=lambda: envdump.run())

@app.after_request
def after_request(response):
    timestamp = strftime('[%Y-%b-%d %H:%M]')
    logger.info('%s %s %s %s %s %s', timestamp, request.remote_addr, request.method, request.scheme, request.full_path, response.status)
    return response

@app.before_request
def log_request_info():
    logger.info(f'Received {request.method} request on {request.path}')
    logger.debug(f'Headers: {request.headers}')
    logger.debug(f'Body: {request.get_data()}')

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"An error occurred: {str(e)}", exc_info=True)
    return jsonify(error=str(e)), 500

@app.route("/", methods=["GET"])
def hello_world():
    return jsonify({"body": "Hello World"}), 200

@app.route("/predict_test", methods=["GET"])
def predict_test():
    input_directory = Path("./input")
    if not input_directory.exists():
        return jsonify({"error": "input directory not found"}), 404

    output_directory = Path("./output")
    output_directory.mkdir(exist_ok=True)

    filepaths = input_directory.glob("*.json")
    logger.debug(f"Filepaths: {filepaths}")

    count = 0
    for filepath in filepaths:
        logger.debug(f"Processing file: {filepath}")
        with open(filepath, "r") as file:
            json_input = json.load(file)
        if not isinstance(json_input, list):
            logger.error(
                f"Invalid JSON input. Must be a list of reports. {json_input}"
            )
            continue
        logger.info(f"Number of new reports in {filepath}: {len(json_input)}")

        for i, report_item in enumerate(json_input[:5]):
            if "RPT_TEXT" not in report_item:
                logger.error("RPT_TEXT not found in report_item.")
                continue
            try:
                report = report_item["RPT_TEXT"]
                report_with_table = get_report_with_table(report)
                logger.debug(f"Report with table: {report_with_table}")
                report_item["RPT_TEXT"] = report_with_table
                count += 1
            except Exception as e:
                logger.error(f"Error in processing report {i}: {e}")

        logger.info(
            f"{filepath}: Done predicting new reports. Number of processed reports: {count}"
        )

        fileout = output_directory / filepath.name
        with fileout.open("w") as file:
            json.dump(json_input, file, indent=4)

    return {
        "statusCode": 200,
        "body": json.dumps({"processed reports count": count}, indent=4),
    }


@app.route("/predict", methods=["GET"])
def process_files():
    s3 = boto3.resource("s3")
    bucket_name = os.getenv("S3_BUCKET_NAME")
    bucket = s3.Bucket(bucket_name)
    objects_list = bucket.objects.all()

    count = 0
    for obj in objects_list:
        obj_key = obj.key
        obj_body = obj.get()["Body"].read().decode("utf-8")
        logger.debug(f"File {obj_key} read from bucket.")

        try:
            # Load the JSON content
            json_input = json.loads(obj_body)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {obj_key}: {e}")
            continue
        if not isinstance(json_input, list):
            logger.error(
                f"Invalid JSON input. Must be a list of reports. {json_input}"
            )
            continue
        logger.info("Number of new reports: " + str(len(json_input)))

        for i, report_item in enumerate(json_input):
            if "RPT_TEXT" not in report_item:
                logger.error("RPT_TEXT not found in report_item.")
                continue
            try:
                report = report_item["RPT_TEXT"]
                report_with_table = get_report_with_table(report)
                report_item["RPT_TEXT"] = report_with_table
                count += 1
            except Exception as e:
                logger.error(f"Error in processing report {i}: {e}")
        logger.info(
            f"Done predicting new reports. Number of processed reports: {count}"
        )

        updated_json = json.dumps(json_input, indent=4).encode("utf-8")
        bucket.put_object(
            Body=updated_json,
            Key=obj_key,  # WARN: save to same key or different key?
        )

    return {
        "statusCode": 200,
        "body": json.dumps({"processed reports count": count}, indent=4),
    }


def main():
    try:
        logger.info("Loading models...")
        load_models()
    except Exception as e:
        logger.error(f"Error in loading models: {e}")
        return

    try:
        logger.info("Loading environment variables...")
        load_dotenv()
        logger.debug(
            f"S3 bucket name environment variable: {os.getenv('S3_BUCKET_NAME')}"
        )
    except Exception as e:
        logger.error("Error in loading environment variables: {e}")
        return

    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)
    
if __name__ == "__main__":
    main()

