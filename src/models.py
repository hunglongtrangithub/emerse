from pathlib import Path
from pydantic import BaseModel
from typing import Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from .config import logger, MAX_LENGTH


class ModelConfig(BaseModel):
    name: str
    model: Any
    tokenizer: Any


class PathologyPrediction(BaseModel):
    field: str
    field_name: str
    value: list
    max_prob: list


class PathologyModelConfig(ModelConfig):
    num_labels: int


class PathologyModels(BaseModel):
    pat: PathologyModelConfig
    lat: PathologyModelConfig
    gra: PathologyModelConfig
    lym: PathologyModelConfig


class PathologyModelRegistry:
    def __init__(self, models_dir: str, device: torch.device):
        if not Path(models_dir).exists():
            raise FileNotFoundError(
                f"Pathology models directory not found: {models_dir}"
            )
        self.models_dir = Path(models_dir)
        self.device = device

    def _load_model(
        self,
        name: str,
        checkpoint: Path,
        num_labels: int,
        ignore_mismatched_sizes: bool = False,
    ) -> PathologyModelConfig:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        ).to(self.device)

        logger.info(f"{name} model loaded successfully from {checkpoint}")
        return PathologyModelConfig(
            name=name,
            model=model,
            tokenizer=tokenizer,
            num_labels=num_labels,
        )

    def load_models(self):
        logger.info(f"Loading models on device: {self.device}")

        self.models = PathologyModels(
            pat=self._load_model(
                name="Pathology",
                checkpoint=self.models_dir / "pat_BigBird_mimic_3_path_3_epoch30_new",
                num_labels=2,
                ignore_mismatched_sizes=True,
            ),
            lat=self._load_model(
                name="Laterality",
                checkpoint=self.models_dir / "lat_BigBird_mimic_3_path_3_epoch30",
                num_labels=5,
            ),
            gra=self._load_model(
                name="Grade",
                checkpoint=self.models_dir / "gra_BigBird_mimic_3_path_3_epoch30",
                num_labels=5,
            ),
            lym=self._load_model(
                name="Lymph Node Metastasis",
                checkpoint=self.models_dir / "lym_BigBird_mimic_3_path_3_epoch30",
                num_labels=4,
                ignore_mismatched_sizes=True,
            ),
        )

    def model_predict(
        self,
        input_texts: list[str],
        model_config: PathologyModelConfig,
    ) -> tuple[list[str], list[float]]:
        if not input_texts:
            return [], []

        model, tokenizer = model_config.model, model_config.tokenizer

        # Tokenization
        inputs = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        # Model inference
        with torch.no_grad():
            outputs = model(**inputs)
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

    def predict(self, report_texts: list[str]) -> list[PathologyPrediction]:
        if self.models is None:
            logger.info("Models are not loaded. Please load models first.")
            return []

        # Perform predictions for each task
        predicted_labels_pat, max_probs_pat = self.model_predict(
            report_texts, self.models.pat
        )
        predicted_labels_lat, max_probs_lat = self.model_predict(
            report_texts, self.models.lat
        )
        predicted_labels_gra, max_probs_gra = self.model_predict(
            report_texts, self.models.gra
        )
        predicted_labels_lym, max_probs_lym = self.model_predict(
            report_texts, self.models.lym
        )

        # Return structured predictions
        return [
            PathologyPrediction(
                field="Pathology",
                field_name="PAT",
                value=predicted_labels_pat,
                max_prob=max_probs_pat,
            ),
            PathologyPrediction(
                field="Laterality",
                field_name="LAT",
                value=predicted_labels_lat,
                max_prob=max_probs_lat,
            ),
            PathologyPrediction(
                field="Grade",
                field_name="GRA",
                value=predicted_labels_gra,
                max_prob=max_probs_gra,
            ),
            PathologyPrediction(
                field="Lymph Node Metastasis",
                field_name="LYM",
                value=predicted_labels_lym,
                max_prob=max_probs_lym,
            ),
        ]


class BERT(nn.Module):
    def __init__(self, num_ner_labels, model_name):
        super(BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        # For NER
        self.ner_dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.ner_output = nn.Linear(self.bert.config.hidden_size, num_ner_labels)

    def forward(self, input_ids=None):
        embeddings = self.bert(input_ids)

        sequence_output = embeddings[0]
        sequence_output = self.ner_dropout(sequence_output)
        logits = self.ner_output(sequence_output)
        return logits


class MobilityModelConfig(ModelConfig):
    entity: str
    tag_list: list[str]


class MobilityModels(BaseModel):
    action: MobilityModelConfig
    assistant: MobilityModelConfig
    mobility: MobilityModelConfig
    quantification: MobilityModelConfig


class MobilityPrediction(BaseModel):
    tokenized_input: list[str]
    output_tags: list[str]
    entity_indexes: list[list[int]]
    extracted_entities: list[str]


class MobilityModelRegistry:
    def __init__(self, models_dir: str, device: torch.device):
        if not Path(models_dir).exists():
            raise FileNotFoundError(
                f"Mobility models directory not found: {models_dir}"
            )
        self.models_dir = Path(models_dir)
        self.device = device

    def _load_model(self, entity: str) -> MobilityModelConfig:
        pretrained_model_name = "UFNLP/Gatortron-base"
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        model = BERT(num_ner_labels=3, model_name=pretrained_model_name)
        model.load_state_dict(
            torch.load(self.models_dir / entity / "best_model_state.bin")
        )
        model.to(self.device)
        model.eval()

        return MobilityModelConfig(
            name=entity,
            entity=entity,
            model=model,
            tokenizer=tokenizer,
            tag_list=[f"B-{entity}", f"I-{entity}", "O"],
        )

    def load_models(self):
        self.models = MobilityModels(
            action=self._load_model("Action"),
            assistant=self._load_model("Assistant"),
            mobility=self._load_model("Mobility"),
            quantification=self._load_model("Quantification"),
        )

    def model_extract_entity_indexes(
        self, input_texts: list[str], model_config: MobilityModelConfig
    ) -> list[MobilityPrediction]:
        """
        Extract entity indexes from a list of texts using a trained BERT-based model.
        """
        results = []

        for sentence in input_texts:
            encoding = model_config.tokenizer(
                sentence, return_tensors="pt", truncation=True, padding=True
            )
            input_ids = encoding["input_ids"].to(self.device)
            outputs = model_config.model(input_ids)
            _, preds = torch.max(outputs[0], dim=1)

            tokens = model_config.tokenizer.convert_ids_to_tokens(
                input_ids.to("cpu").numpy()[0]
            )

            # Initialize character positions for each token
            new_tokens, output_tags, token_char_positions = [], [], []
            char_offset = 0

            for token, tag_idx in zip(tokens, preds):
                if token.startswith("##"):
                    # For subword tokens, append to the previous token and adjust the character position
                    new_tokens[-1] += token[2:]
                    token_char_positions[-1][1] += len(
                        token[2:]
                    )  # Extend the end position
                    char_offset = token_char_positions[-1][1]
                else:
                    if token not in ["[PAD]", "[CLS]", "[SEP]"]:
                        # Find the start and end position in the original sentence
                        while sentence[char_offset] == " ":
                            char_offset += 1  # Skip any spaces

                        start_position = char_offset
                        end_position = start_position + len(token)

                        output_tags.append(model_config.tag_list[tag_idx])
                        new_tokens.append(token)
                        token_char_positions.append([start_position, end_position])

                        char_offset = end_position  # Move to the next character offset

            # Extract entity indexes
            current_positions = []
            for i, tag in enumerate(output_tags):
                if tag == f"B-{model_config.entity}":  # Start of an entity
                    current_positions.append(token_char_positions[i])
                elif tag == f"I-{model_config.entity}":  # Continuation of an entity
                    current_positions[-1][1] = token_char_positions[i][
                        1
                    ]  # Extend the end position

            # Append results for this sentence
            results.append(
                MobilityPrediction(
                    tokenized_input=new_tokens,
                    output_tags=output_tags,
                    entity_indexes=current_positions,
                    extracted_entities=[
                        sentence[idx[0] : idx[1]] for idx in current_positions
                    ],
                )
            )

        return results

    def extract_entity_indexes(
        self, report_texts: list[str]
    ) -> list[list[MobilityPrediction]]:
        if self.models is None:
            logger.info("Models are not loaded. Please load models first.")
            return []
        return [
            self.model_extract_entity_indexes(report_texts, self.models.action),
            self.model_extract_entity_indexes(report_texts, self.models.mobility),
            self.model_extract_entity_indexes(report_texts, self.models.assistant),
            self.model_extract_entity_indexes(report_texts, self.models.quantification),
        ]


class ModelRegistry:
    def __init__(self, pathology_models_dir: str, mobility_models_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pathology_registry = PathologyModelRegistry(
            pathology_models_dir, self.device
        )
        self.mobility_registry = MobilityModelRegistry(mobility_models_dir, self.device)
        self.models_loaded = False

    def load_models(self):
        self.pathology_registry.load_models()
        self.mobility_registry.load_models()
        self.models_loaded = True

    def check_model_health(self, model_config: ModelConfig):
        """Perform a basic health check for a model by running a dummy inference."""
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

    def check_all_models_health(self) -> tuple[bool, list[str]]:
        all_models = {
            **self.pathology_registry.models.model_dump(),
            **self.mobility_registry.models.model_dump(),
        }
        messages = []
        is_all_models_healthy = True
        for model_config in all_models.values():
            is_model_healthy, message = self.check_model_health(model_config)
            is_all_models_healthy = is_all_models_healthy and is_model_healthy
            messages.append(message)

        return is_all_models_healthy, messages
