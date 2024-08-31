import json
import os
import glob
from pathlib import Path
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

if os.getenv("DEBUG"):
    print("Debug mode enabled")
else:
    print("Debug mode disabled")


def test_path():
    path = Path("./input")
    if path.exists():
        print("Path exists")
    filepaths = path.glob("*.json")
    filepath = next(filepaths)
    print(filepath)


def get_json_files(directory):
    return glob.glob(os.path.join(directory, "*.json"))


def test_soup():
    html = """
    <html><body><body><html>
    """
    soup = BeautifulSoup(html, "html.parser")
    print(soup.body)


def test_batch():
    from run import load_models, get_reports_with_table

    load_models()
    reports = json.loads(open("input/test_batch.json").read())
    reports = get_reports_with_table(reports)
    json.dump(reports, open("output/test_batch.json", "w"), indent=2)


if __name__ == "__main__":
    test_batch()
