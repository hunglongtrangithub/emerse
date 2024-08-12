import os
import glob
from pathlib import Path
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
if os.getenv("DEBUG"):
    print("Debug mode enabled")
else:
    print("Debug mode disabled")

def test_path():
    path = Path("./input")
    if path.exists():
        print("Path exists")
    filepaths = path.glob('*.json')
    filepath = next(filepaths)

def get_json_files(directory):
    return glob.glob(os.path.join(directory, '*.json'))

def test_soup():
    html = """
    <html><body><body><html>
    """
    soup = BeautifulSoup(html, "html.parser")
    print(soup.body)

def test_batch():
    # use a sentiment analysis model
    checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    batch_inputs = ["I love you", "I hate you"]
    inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    # map to sentiment labels
    print(outputs.logits.shape)
    sentiment = outputs.logits.argmax(dim=1)
    print(sentiment)
if __name__ == "__main__":
    test_batch()
