import os
from pathlib import Path
import json
import logging

from dotenv import load_dotenv
from flask import Flask, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import boto3
import html2text
from bs4 import BeautifulSoup
from yattag import Doc

# Todo: Discuss with IT on a protocol to find out which files have been processed,
# and contain the results, which have failed and probably why, and intergrate it into the code.

logger = logging.getLogger(__name__)

max_length = 4096 # Maximum length of input sequence to cap


def load_models(device):
    global \
    tokenizer_pat, \
    model_pat, \
    tokenizer_lat, \
    model_lat, \
    tokenizer_gra, \
    model_gra, \
    tokenizer_lym, \
    model_lym
    model_name_pat = "./models/pat_BigBird_mimic_3_path_3_epoch30_new"
    tokenizer_pat = AutoTokenizer.from_pretrained(model_name_pat)
    model_pat = AutoModelForSequenceClassification.from_pretrained(
        model_name_pat, num_labels=2, ignore_mismatched_sizes=True
    )
    model_pat = model_pat.to(device)
    logger.info("Pathology prediction model loaded successfully: " + model_name_pat)

    model_name_lat = "./models/lat_BigBird_mimic_3_path_3_epoch30"
    tokenizer_lat = AutoTokenizer.from_pretrained(model_name_lat)
    model_lat = AutoModelForSequenceClassification.from_pretrained(
        model_name_lat, num_labels=5
    )
    model_lat = model_lat.to(device)
    logger.info("Laterality prediction model loaded successfully: " + model_name_lat)

    model_name_gra = "./models/gra_BigBird_mimic_3_path_3_epoch30"
    tokenizer_gra = AutoTokenizer.from_pretrained(model_name_gra)
    model_gra = AutoModelForSequenceClassification.from_pretrained(
        model_name_gra, num_labels=5
    )
    model_gra = model_gra.to(device)
    logger.info("Grade prediction model loaded successfully: " + model_name_gra)

    model_name_lym = "./models/lym_BigBird_mimic_3_path_3_epoch30"
    tokenizer_lym = AutoTokenizer.from_pretrained(model_name_lym)
    model_lym = AutoModelForSequenceClassification.from_pretrained(
        model_name_lym, num_labels=4, ignore_mismatched_sizes=True
    )
    model_lym = model_lym.to(device)
    logger.info(
        "Lymph node metastasis prediction model loaded successfully: " + model_name_lym
    )
    return


def pat_prediction(input_text):
    # Tokenize input sequence with truncation
    inputs = tokenizer_pat(
        input_text, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model_pat(**inputs)
        logits = outputs.logits

    # Apply softmax to get confidence scores
    probs = torch.nn.functional.softmax(logits, dim=1)
    max_prob1, predicted_label_id = torch.max(probs, dim=1)

    predicted_label1 = model_pat.config.id2label[predicted_label_id.item()]

    return predicted_label1, max_prob1.item()


def lat_prediction(input_text):
    # Tokenize input sequence with truncation
    inputs = tokenizer_lat(
        input_text, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model_lat(**inputs)
        logits = outputs.logits

    # Apply softmax to get confidence scores
    probs = torch.nn.functional.softmax(logits, dim=1)
    max_prob1, predicted_label_id = torch.max(probs, dim=1)

    predicted_label1 = model_lat.config.id2label[predicted_label_id.item()]

    return predicted_label1, max_prob1.item()


def gra_prediction(input_text):
    # Tokenize input sequence with truncation
    inputs = tokenizer_gra(
        input_text, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model_gra(**inputs)
        logits = outputs.logits

    # Apply softmax to get confidence scores
    probs = torch.nn.functional.softmax(logits, dim=1)
    max_prob1, predicted_label_id = torch.max(probs, dim=1)

    predicted_label1 = model_gra.config.id2label[predicted_label_id.item()]

    return predicted_label1, max_prob1.item()


def lym_prediction(input_text):
    # Tokenize input sequence with truncation
    inputs = tokenizer_lym(
        input_text, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model_lym(**inputs)
        logits = outputs.logits

    # Apply softmax to get confidence scores
    probs = torch.nn.functional.softmax(logits, dim=1)
    max_prob1, predicted_label_id = torch.max(probs, dim=1)

    predicted_label1 = model_lym.config.id2label[predicted_label_id.item()]

    return predicted_label1, max_prob1.item()


def predict_and_create_output(report):
    predicted_label_pat, max_prob_pat = pat_prediction(report)
    predicted_label_lat, max_prob_lat = lat_prediction(report)
    predicted_label_gra, max_prob_gra = gra_prediction(report)
    predicted_label_lym, max_prob_lym = lym_prediction(report)

    json_output = {
        "predicted_label_pat": predicted_label_pat,
        "max_prob_pat": max_prob_pat,
        "predicted_label_lat": predicted_label_lat,
        "max_prob_lat": max_prob_lat,
        "predicted_label_gra": predicted_label_gra,
        "max_prob_gra": max_prob_gra,
        "predicted_label_lym": predicted_label_lym,
        "max_prob_lym": max_prob_lym,
    }
    html_table = output2table(
        predicted_label_pat,
        predicted_label_lat,
        predicted_label_gra,
        predicted_label_lym,
        max_prob_pat,
        max_prob_lat,
        max_prob_gra,
        max_prob_lym,
    )
    return json_output, html_table



# TODO: @HUNG make this more forward compatible (use dictionary with datastructure containing: field, field name, value, max_prob)
def output2table(pat, lat, gra, lym, pat_prob, lat_prob, gra_prob, lym_prob):
    tableString = """<table>
                <tr>
                <th></th>
                <th>Pat</th>
                <th>Lat</th>
                <th>Gra</th>
                <th>Lym</th>
            </tr>
            <tr>
                <th>Value</th>
                <td><field name="PAT" type="string">{PAT}</field></td>
                <td><field name="LAT" type="string">{LAT}</field></td>
                <td><field name="GRA" type="string">{GRA}</field></td>
                <td><field name="LYM" type="string">{LYM}</field></td>
        </tr>
        <tr>
            <th>Max Prob</th>
            <td><field name="PAT_prob" type="string">{PAT_prob}</field></td>
            <td><field name="LAT_prob" type="string">{LAT_prob}</field></td>
            <td><field name="GRA_prob" type="string">{GRA_prob}</field></td>
            <td><field name="LYM_prob" type="string">{LYM_prob}</field></td>
        </tr>
    </table>""".format(
        PAT=pat,
        LAT=lat,
        GRA=gra,
        LYM=lym,
        PAT_prob=pat_prob,
        LAT_prob=lat_prob,
        GRA_prob=gra_prob,
        LYM_prob=lym_prob,
    )

    return tableString


def html_2_text(html_input):
    h = html2text.HTML2Text()
    h.ignore_links = True

    return h.handle(html_input)
def get_report_with_table(report_html: str):
    report_html = html_2_text(report_html)
    _, html_table = predict_and_create_output(report_html)
    end_of_body_index = report_html.rfind("</body>")
    report_with_table = (
        report_html[:end_of_body_index]
        + html_table
        + report_html[end_of_body_index:]
    )
    return report_with_table
    soup = BeautifulSoup(report_html, "html.parser")
    if soup.body is None:
        raise ValueError("No body tag found in the report.")
    report = soup.body.text

    doc, tag, text = Doc().tagtext()
    with tag("body"):
        text(soup.body)
        with tag("table"):
            with tag("tr"):
                text("Predicted Label")
                text("Max Probability")
            with tag("tr"):
                text("Pathology")
                text(predicted_label_pat)
                text(max_prob_pat)
            with tag("tr"):
                text("Laterality")
                text(predicted_label_lat)
                text(max_prob_lat)
            with tag("tr"):
                text("Grade")
                text(predicted_label_gra)
                text(max_prob_gra)
            with tag("tr"):
                text("Lymph Node Metastasis")
                text(predicted_label_lym)
                text(max_prob_lym)
    return doc.getvalue()
    


app = Flask(__name__)


@app.route("/health_check", methods=["GET"])
# @HUNG ToDo: write a proper health check for flask server, check environment variables, check models , ...
# https://py-healthcheck.readthedocs.io/en/stable/flask.html
def health_check():
    return jsonify({"body": "Simple health check"}), 200


@app.route("/predict", methods=["GET"])
def process_value():
    try:
        if debug:
            input_directory = Path("./input")
            if not input_directory.exists():
                return jsonify({"error": "Directory not found"}), 404

            output_directory = Path("./output")
            output_directory.mkdir(exist_ok=True)

            filepaths = input_directory.glob('*.json')
            logger.debug(f"Filepaths: {filepaths}")

            count = 0
            for filepath in filepaths:
                with open(filepath, "r") as file:
                    json_input = json.load(file)
                logger.info(f"Number of new reports: {len(json_input)}")

                for report_item in json_input:
                    if "RPT_TEXT" not in report_item:
                        logger.error("RPT_TEXT not found in report_item.")
                        continue
                    count += 1
                    report = report_item["RPT_TEXT"]
                    report_with_table = get_report_with_table(report)
                    report_item["RPT_TEXT"] = report_with_table
                logger.info(f"Done predicting new reports. Number of processed reports: {count}")
                
                fileout = output_directory / filepath.name
                with fileout.open("w") as file:
                    json.dump(json_input, file, indent=4)

            return {"statusCode": 200, "body": json.dumps({"processed reports count": count}, indent=4)}

        else:
            s3 = boto3.resource("s3")
            bucket_name = os.getenv("S3_BUCKET_NAME")
            bucket = s3.Bucket(bucket_name)
            objects_list = bucket.objects.all()
            
            count = 0
            for obj in objects_list:
                obj_key = obj.key
                obj_body = obj.get()["Body"].read().decode("utf-8")
                try:
                    # Load the JSON content
                    json_input = json.loads(obj_body)
                    logger.debug(f"File {obj_key} read from bucket.")
                    logger.info("Number of new reports: " + str(len(json_input)))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON from {obj_key}: {e}")
                    continue
                logger.debug(f"File {obj_key} read from bucket.")

                for report_item in json_input:
                    if "RPT_TEXT" not in report_item:
                        logger.error("RPT_TEXT not found in report_item.")
                        continue
                    count += 1
                    report = report_item["RPT_TEXT"]
                    report_with_table = get_report_with_table(report)
                    report_item["RPT_TEXT"] = report_with_table
                logger.info(f"Done predicting new reports. Number of processed reports: {count}")

                updated_json = json.dumps(json_input, indent=4).encode("utf-8")
                bucket.put_object(
                    Body=updated_json,
                    Key=obj_key,  # WARN: save to same key or different key?
                )
        return {"statusCode": 200, "body": json.dumps({"processed reports count": count}, indent=4)}

    except FileNotFoundError:
        return jsonify({"error": "Directory not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    except:
        return jsonify({"error": "Invalid type parameter"}), 400


if __name__ == "__main__":
    debug = os.getenv("DEBUG") or False

    if debug:
        logging.basicConfig(filename="./log/nlp_app.log", level=logging.DEBUG)
    else:
        logging.basicConfig(filename="./log/nlp_app.log", level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device recognized:", device)
    try:
        load_models(device)
    except Exception as e:
        logger.error(f"Error in loading models: {e}")

    try:
        load_dotenv()
        logger.debug(
            f"S3 bucket name environment variable: {os.getenv('S3_BUCKET_NAME')}"
        )
    except Exception as e:
        logger.error("Error in loading environment variables: {e}")

    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)

