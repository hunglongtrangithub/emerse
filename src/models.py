from pydantic import BaseModel
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import logger, MAX_LENGTH


class Prediction(BaseModel):
    field: str
    field_name: str
    value: list
    max_prob: list


class ModelConfig(BaseModel):
    name: str
    checkpoint: str
    model: Any
    tokenizer: Any
    additional_info: dict[str, Any] = {}


class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False

    def load_models(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading models on device: {device}")

        model_name_pat = "./models/pat_BigBird_mimic_3_path_3_epoch30_new"
        tokenizer_pat = AutoTokenizer.from_pretrained(model_name_pat)
        model_pat = AutoModelForSequenceClassification.from_pretrained(
            model_name_pat, num_labels=2, ignore_mismatched_sizes=True
        ).to(device)
        self.pat = ModelConfig(
            name="Pathology",
            checkpoint=model_name_pat,
            model=model_pat,
            tokenizer=tokenizer_pat,
            additional_info={"num_labels": 2},
        )
        logger.info(f"Pathology prediction model loaded successfully: {model_name_pat}")

        model_name_lat = "./models/lat_BigBird_mimic_3_path_3_epoch30"
        tokenizer_lat = AutoTokenizer.from_pretrained(model_name_lat)
        model_lat = AutoModelForSequenceClassification.from_pretrained(
            model_name_lat, num_labels=5
        ).to(device)
        self.lat = ModelConfig(
            name="Laterality",
            checkpoint=model_name_lat,
            model=model_lat,
            tokenizer=tokenizer_lat,
            additional_info={"num_labels": 5},
        )
        logger.info(
            f"Laterality prediction model loaded successfully: {model_name_lat}"
        )

        model_name_gra = "./models/gra_BigBird_mimic_3_path_3_epoch30"
        tokenizer_gra = AutoTokenizer.from_pretrained(model_name_gra)
        model_gra = AutoModelForSequenceClassification.from_pretrained(
            model_name_gra, num_labels=5
        ).to(device)
        self.gra = ModelConfig(
            name="Grade",
            checkpoint=model_name_gra,
            model=model_gra,
            tokenizer=tokenizer_gra,
            additional_info={"num_labels": 5},
        )
        logger.info(f"Grade prediction model loaded successfully: {model_name_gra}")

        model_name_lym = "./models/lym_BigBird_mimic_3_path_3_epoch30"
        tokenizer_lym = AutoTokenizer.from_pretrained(model_name_lym)
        model_lym = AutoModelForSequenceClassification.from_pretrained(
            model_name_lym, num_labels=4, ignore_mismatched_sizes=True
        ).to(device)
        self.lym = ModelConfig(
            name="Lymph Node Metastasis",
            checkpoint=model_name_lym,
            model=model_lym,
            tokenizer=tokenizer_lym,
            additional_info={"num_labels": 4},
        )
        logger.info(
            f"Lymph node metastasis prediction model loaded successfully: {model_name_lym}"
        )
        self.models_loaded = True

    def model_predict(
        self,
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
        if not self.models_loaded:
            logger.info("Models are not loaded. Now loading models...")
            self.load_models()

        # Perform predictions for each task
        predicted_labels_pat, max_probs_pat = self.model_predict(
            report_texts, self.pat.model, self.pat.tokenizer
        )
        predicted_labels_lat, max_probs_lat = self.model_predict(
            report_texts, self.lat.model, self.lat.tokenizer
        )
        predicted_labels_gra, max_probs_gra = self.model_predict(
            report_texts, self.gra.model, self.gra.tokenizer
        )
        predicted_labels_lym, max_probs_lym = self.model_predict(
            report_texts, self.lym.model, self.lym.tokenizer
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

    def check_model_health(self, model_config: ModelConfig):
        """Perform a basic health check for a model by running a dummy inference."""
        try:
            test_input = ["Test"]  # Dummy input for health check
            try:
                inputs = model_config.tokenizer(
                    test_input,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    _ = model_config.model(**inputs)
                return True, f"Model '{model_config.name}' is healthy"
            except Exception as e:
                return False, f"Model '{model_config.name}' is unhealthy: {str(e)}"
        except Exception as e:
            return False, str(e)


# Singleton instance of ModelRegistry
model_registry = ModelRegistry()
