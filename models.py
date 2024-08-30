from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import logger, MAX_LENGTH

class Prediction(BaseModel):
    field: str
    field_name: str
    value: list
    max_prob: list

class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_pat = None
        self.model_pat = None
        self.tokenizer_lat = None
        self.model_lat = None
        self.tokenizer_gra = None
        self.model_gra = None
        self.tokenizer_lym = None
        self.model_lym = None   

    def load_models(self):
        logger.info(f"Loading models on device: {self.device}")

        model_name_pat = "./models/pat_BigBird_mimic_3_path_3_epoch30_new"
        self.tokenizer_pat = AutoTokenizer.from_pretrained(model_name_pat)
        self.model_pat = AutoModelForSequenceClassification.from_pretrained(
            model_name_pat, num_labels=2, ignore_mismatched_sizes=True
        ).to(self.device)
        logger.info(f"Pathology prediction model loaded successfully: {model_name_pat}")

        model_name_lat = "./models/lat_BigBird_mimic_3_path_3_epoch30"
        self.tokenizer_lat = AutoTokenizer.from_pretrained(model_name_lat)
        self.model_lat = AutoModelForSequenceClassification.from_pretrained(
            model_name_lat, num_labels=5
        ).to(self.device)
        logger.info(f"Laterality prediction model loaded successfully: {model_name_lat}")

        model_name_gra = "./models/gra_BigBird_mimic_3_path_3_epoch30"
        self.tokenizer_gra = AutoTokenizer.from_pretrained(model_name_gra)
        self.model_gra = AutoModelForSequenceClassification.from_pretrained(
            model_name_gra, num_labels=5
        ).to(self.device)
        logger.info(f"Grade prediction model loaded successfully: {model_name_gra}")

        model_name_lym = "./models/lym_BigBird_mimic_3_path_3_epoch30"
        self.tokenizer_lym = AutoTokenizer.from_pretrained(model_name_lym)
        self.model_lym = AutoModelForSequenceClassification.from_pretrained(
            model_name_lym, num_labels=4, ignore_mismatched_sizes=True
        ).to(self.device)
        logger.info(f"Lymph node metastasis prediction model loaded successfully: {model_name_lym}")

    def model_predict(self, input_texts: list[str], model, tokenizer, tokenizer_kwargs={}, model_kwargs={}):
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
        ).to(self.device)

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

    def predict(self, report_texts: list[str]) -> list[Prediction]:
        # Perform predictions for each task
        predicted_labels_pat, max_probs_pat = self.model_predict(
            report_texts, self.model_pat, self.tokenizer_pat
        )
        predicted_labels_lat, max_probs_lat = self.model_predict(
            report_texts, self.model_lat, self.tokenizer_lat
        )
        predicted_labels_gra, max_probs_gra = self.model_predict(
            report_texts, self.model_gra, self.tokenizer_gra
        )
        predicted_labels_lym, max_probs_lym = self.model_predict(
            report_texts, self.model_lym, self.tokenizer_lym
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
