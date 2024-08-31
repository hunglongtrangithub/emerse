from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import get_logger, MAX_LENGTH

logger = get_logger(__name__)


class Prediction(BaseModel):
    field: str
    field_name: str
    value: list
    max_prob: list


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
    ).to(device)
    logger.info(f"Pathology prediction model loaded successfully: {model_name_pat}")

    model_name_lat = "./models/lat_BigBird_mimic_3_path_3_epoch30"
    tokenizer_lat = AutoTokenizer.from_pretrained(model_name_lat)
    model_lat = AutoModelForSequenceClassification.from_pretrained(
        model_name_lat, num_labels=5
    ).to(device)
    logger.info(f"Laterality prediction model loaded successfully: {model_name_lat}")

    model_name_gra = "./models/gra_BigBird_mimic_3_path_3_epoch30"
    tokenizer_gra = AutoTokenizer.from_pretrained(model_name_gra)
    model_gra = AutoModelForSequenceClassification.from_pretrained(
        model_name_gra, num_labels=5
    ).to(device)
    logger.info(f"Grade prediction model loaded successfully: {model_name_gra}")

    model_name_lym = "./models/lym_BigBird_mimic_3_path_3_epoch30"
    tokenizer_lym = AutoTokenizer.from_pretrained(model_name_lym)
    model_lym = AutoModelForSequenceClassification.from_pretrained(
        model_name_lym, num_labels=4, ignore_mismatched_sizes=True
    ).to(device)
    logger.info(
        f"Lymph node metastasis prediction model loaded successfully: {model_name_lym}"
    )


def model_predict(
    input_texts: list[str],
    model,
    tokenizer,
    tokenizer_kwargs={},
    model_kwargs={},
):
    if not input_texts:
        return [], []

    # Tokenization
    inputs = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        **tokenizer_kwargs,
    ).to(device)

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs, **model_kwargs)
        logits = outputs.logits

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=1)
    max_probs, predicted_label_ids = torch.max(probs, dim=1)
    max_probs = max_probs.tolist()
    predicted_label_ids = predicted_label_ids.tolist()
    predicted_labels = [
        model.config.id2label[label_id] for label_id in predicted_label_ids
    ]

    return predicted_labels, max_probs


def predict(report_texts: list[str]) -> list[Prediction]:
    # Perform predictions for each task
    predicted_labels_pat, max_probs_pat = model_predict(
        report_texts, model_pat, tokenizer_pat
    )
    predicted_labels_lat, max_probs_lat = model_predict(
        report_texts, model_lat, tokenizer_lat
    )
    predicted_labels_gra, max_probs_gra = model_predict(
        report_texts, model_gra, tokenizer_gra
    )
    predicted_labels_lym, max_probs_lym = model_predict(
        report_texts, model_lym, tokenizer_lym
    )

    # Return structured predictions
    return [
        Prediction(
            field="Pathology",
            field_name="PAT",
            value=predicted_labels_pat,
            max_prob=max_probs_pat,
        ),
        Prediction(
            field="Laterality",
            field_name="LAT",
            value=predicted_labels_lat,
            max_prob=max_probs_lat,
        ),
        Prediction(
            field="Grade",
            field_name="GRA",
            value=predicted_labels_gra,
            max_prob=max_probs_gra,
        ),
        Prediction(
            field="Lymph Node Metastasis",
            field_name="LYM",
            value=predicted_labels_lym,
            max_prob=max_probs_lym,
        ),
    ]
