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
from jinja2 import Environment, FileSystemLoader

# Load the Jinja2 environment
template_dir = Path(__file__).parent / "templates"
env = Environment(loader=FileSystemLoader(template_dir))
template = env.get_template("report_template.html")
# print(template.render())

MODE = os.getenv("MODE") or "dev"
MAX_LENGTH = 4096  # Maximum length of input sequence to truncate
BATCH_SIZE = 32  # Batch size for prediction

if MODE == "debug":
    logging.basicConfig(level=logging.DEBUG)
elif MODE == "dev":
    logging.basicConfig(level=logging.INFO)
else:
    Path("./log").mkdir(exist_ok=True)
    logging.basicConfig(filename="./log/nlp_app.log", level=logging.INFO)

logger = logging.getLogger(__name__)
logger.info(f"Mode: {MODE}")
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


def model_predict(
    input_texts: list[str],
    model,
    tokenizer,
    device,
    tokenizer_kwargs={},
    model_kwargs={},
):
    if not input_texts:
        return [], []
    # Read about truncation & padding behaviors here: https://huggingface.co/docs/transformers/pad_truncation
    inputs = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        **tokenizer_kwargs,
    ).to(device)

    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs, **model_kwargs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=1)
    max_probs, predicted_label_ids = torch.max(probs, dim=1)
    max_probs, predicted_label_ids = max_probs.tolist(), predicted_label_ids.tolist()
    predicted_label = [
        model.config.id2label[label_id] for label_id in predicted_label_ids
    ]

    return predicted_label, max_probs


def predict(report_texts):
    # all of these lists have the same length
    # add other generation arguments as needed here
    predicted_labels_pat, max_probs_pat = model_predict(
        report_texts, model_pat, tokenizer_pat, device
    )
    predicted_labels_lat, max_probs_lat = model_predict(
        report_texts, model_lat, tokenizer_lat, device
    )
    predicted_labels_gra, max_probs_gra = model_predict(
        report_texts, model_gra, tokenizer_gra, device
    )
    predicted_labels_lym, max_probs_lym = model_predict(
        report_texts, model_lym, tokenizer_lym, device
    )

    return [
        {
            "field": "PAT",
            "field_name": "Pathology",
            "value": predicted_labels_pat,
            "max_prob": max_probs_pat,
        },
        {
            "field": "LAT",
            "field_name": "Laterality",
            "value": predicted_labels_lat,
            "max_prob": max_probs_lat,
        },
        {
            "field": "GRA",
            "field_name": "Grade",
            "value": predicted_labels_gra,
            "max_prob": max_probs_gra,
        },
        {
            "field": "LYM",
            "field_name": "Lymph Node Metastasis",
            "value": predicted_labels_lym,
            "max_prob": max_probs_lym,
        },
    ]


def generate_html_report(report_text, batch_predictions, index):
    """
    Generate an HTML report based on the report text and batch predictions for a specific index.

    :param report_text: The text of the report.
    :param batch_predictions: List of prediction dictionaries.
    :param index: The index of the current item in the batch.
    :return: A string containing the generated HTML.
    """
    # Prepare the data for the template
    predictions = [
        {
            "field_name": prediction_type["field_name"],
            "field": prediction_type["field"],
            "value": prediction_type["value"][index],
            "max_prob": prediction_type["max_prob"][index],
        }
        for prediction_type in batch_predictions
    ]

    # Render the template with the report text and predictions data
    html_content = template.render(report_text=report_text, predictions=predictions)
    return html_content


def is_valid_report(report):
    if "RPT_TEXT" not in report:
        logger.error("RPT_TEXT not found in report.")
        return None
    soup = BeautifulSoup(report["RPT_TEXT"], "html.parser")
    if soup.body is None:
        logger.error("No body tag found in the report.")
        return None
    report_text = soup.body.get_text(strip=True)
    if not report_text or report_text.lower() == "n/a":
        logger.error("No text found in the report.")
        return None
    return report, report_text


def get_reports_with_table(reports):
    """
    Find valid reports, predict the values, and add the predictions to the report as a table.
    Mutate the original reports and return them.
    Invalid reports are returned as is.
    """
    valid_reports = []
    valid_texts = []

    for report in reports:
        report_data = is_valid_report(report)
        if report_data:
            report, report_text = report_data
            valid_reports.append(report)
            valid_texts.append(report_text)

    logger.info(f"Number of valid reports: {len(valid_reports)}")

    for start in range(0, len(valid_reports), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_texts = valid_texts[start:end]
        batch_reports = valid_reports[start:end]

        batch_predictions = predict(batch_texts)

        for i, (report, report_text) in enumerate(zip(batch_reports, batch_texts)):
            report_html = generate_html_report(report_text, batch_predictions, i)
            # Update the original report with the new HTML content
            report["RPT_TEXT"] = report_html

        logger.info(f"Predicted {len(batch_reports)}/{len(valid_reports)} reports")

    return reports


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


@app.route("/predict_test", methods=["GET"])
def predict_test():
    input_directory = Path("./input")
    if not input_directory.exists():
        return jsonify({"error": "input directory not found"}), 404

    output_directory = Path("./output")
    output_directory.mkdir(exist_ok=True)

    filepaths = input_directory.glob("*.json")
    logger.info(f"Filepaths: {filepaths}")

    count = 0
    for filepath in filepaths:
        logger.info(f"Processing file: {filepath}")
        with open(filepath, "r") as file:
            json_input = json.load(file)
        if not isinstance(json_input, list):
            logger.error(f"Invalid JSON input. Must be a list of reports. {json_input}")
            continue
        logger.info(f"Number of new reports in {filepath}: {len(json_input)}")

        json_input = json_input[:5]  # for testing
        try:
            reports = get_reports_with_table(json_input)
        except Exception as e:
            logger.error(f"Error in processing reports: {e}")
            continue

        count += len(reports)
        logger.info(
            f"{filepath}: Done predicting new reports. Number of processed reports: {count}"
        )

        fileout = output_directory / filepath.name
        logger.info(f"Writing to file: {fileout}")
        with fileout.open("w") as file:
            json.dump(json_input, file, indent=4)
        html_out = output_directory / (filepath.stem + ".html")
        for report in reports:
            if "RPT_TEXT" in report:
                with html_out.open("w") as file:
                    file.write(report["RPT_TEXT"])
                break
    return {
        "statusCode": 200,
        "body": json.dumps({"processed reports count": count}, indent=4),
    }


@app.route("/predict", methods=["GET"])
def process_files():
    s3 = boto3.resource("s3")
    bucket_name = os.getenv("S3_BUCKET_NAME")
    logger.info(f"Reading files from bucket: {bucket_name}")
    bucket = s3.Bucket(bucket_name)
    objects_list = bucket.objects.all()

    count = 0
    for obj in objects_list:
        obj_key = obj.key
        obj_body = obj.get()["Body"].read().decode("utf-8")
        logger.info(f"File {obj_key} read from bucket.")

        try:
            # Load the JSON content
            json_input = json.loads(obj_body)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {obj_key}: {e}")
            continue
        if not isinstance(json_input, list):
            logger.error(f"Invalid JSON input. Must be a list of reports. {json_input}")
            continue
        logger.info("Number of new reports: " + str(len(json_input)))

        try:
            reports = get_reports_with_table(json_input)
        except Exception as e:
            logger.error(f"Error in processing reports: {e}")
            continue

        count += len(reports)
        logger.info(
            f"Done predicting new reports. Number of processed reports: {count}"
        )

        updated_json = json.dumps(reports, indent=4).encode("utf-8")
        bucket.put_object(
            Body=updated_json,
            Key=obj_key,  # WARN: save to same key or different key?
        )

    return {
        "statusCode": 200,
        "body": json.dumps({"processed reports count": count}, indent=4),
    }


def main():
    logger.info("Starting NLP application...")
    try:
        logger.info("Loading models...")
        load_models()
    except Exception as e:
        logger.error(f"Error in loading models: {e}")
        return

    try:
        logger.info("Loading environment variables...")
        load_dotenv()
        logger.info(
            f"S3 bucket name environment variable: {os.getenv('S3_BUCKET_NAME')}"
        )
    except Exception as e:
        logger.error("Error in loading environment variables: {e}")
        return

    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
