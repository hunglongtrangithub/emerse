from typing import Literal
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader

from .config import logger, BATCH_SIZE, REPORT_TEXT_COLUMN
from .models import ModelRegistry, PathologyPrediction, MobilityPrediction

# Load the Jinja2 environment
template_dir = "./templates"
env = Environment(loader=FileSystemLoader(template_dir))
pathology_template = env.get_template("pathology_template.html")
mobility_template = env.get_template("mobility_template.html")


def prepare_entities(entity_indexes: dict[str, list[tuple[int, int]]]) -> list[dict]:
    """
    Prepares a flat, sorted list of entities for the mobility template.
    """
    entities = []
    for entity_type, indexes in entity_indexes.items():
        for start, end in indexes:
            entities.append({"start": start, "end": end, "type": entity_type})
    return sorted(entities, key=lambda x: x["start"])


def generate_pathology_html_report(
    report_text: str,
    index: int,
    batch_predictions: list[PathologyPrediction],
) -> str:
    """
    Generate a pathology HTML report.
    """
    predictions = [
        {
            "field_name": prediction.field_name,
            "field": prediction.field,
            "value": prediction.value[index],
            "max_prob": prediction.max_prob[index],
        }
        for prediction in batch_predictions
    ]

    html_content = pathology_template.render(
        report_text=report_text,
        predictions=predictions,
    )
    return html_content


def generate_mobility_html_report(
    report_text: str,
    index: int,
    batch_entity_indexes: dict[str, list[MobilityPrediction]],
) -> str:
    """
    Generate a mobility HTML report.
    """
    entity_indexes = {
        entity_name: [
            (index_pair[0], index_pair[1])
            for index_pair in entity_indexes[index].entity_indexes
        ]
        for entity_name, entity_indexes in batch_entity_indexes.items()
    }
    entities = prepare_entities(entity_indexes)

    html_content = mobility_template.render(
        report_text=report_text,
        entities=entities,
    )
    return html_content


def is_valid_report(report: dict[str, str]) -> tuple[dict[str, str], str] | None:
    if REPORT_TEXT_COLUMN not in report:
        logger.error(f"{REPORT_TEXT_COLUMN} key not found in report.")
        return None
    soup = BeautifulSoup(report[REPORT_TEXT_COLUMN], "html.parser")
    if soup.body is None:
        logger.error("No body tag found in the report.")
        return None
    report_text = soup.body.get_text(strip=True)
    if not report_text or report_text.lower() == "n/a":
        logger.error("No text found in the report.")
        return None
    return report, report_text


PredictType = Literal["pathology", "mobility"]


def process_reports(
    reports: list[dict[str, str]],
    model_registry: ModelRegistry,
    predict_type: PredictType,
) -> list[dict[str, str]]:
    """
    Process reports with either pathology predictions or mobility annotations based on predict_type.

    Args:
        reports: List of report dictionaries
        model_registry: Model registry instance
        predict_type: Type of prediction to perform ("pathology" or "mobility")
        show_output: Whether to show predictions/annotations (defaults to True)

    Returns:
        List of processed reports with HTML content added
    """
    valid_reports = []
    valid_texts = []

    # Validate reports
    for report in reports:
        report_data = is_valid_report(report)
        if report_data:
            report, report_text = report_data
            valid_reports.append(report)
            valid_texts.append(report_text)

    logger.info(f"Number of valid reports: {len(valid_reports)}")

    # Process reports in batches
    for start in range(0, len(valid_reports), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_texts = valid_texts[start:end]
        batch_reports = valid_reports[start:end]

        if predict_type == "pathology":
            # Get pathology predictions
            batch_predictions = model_registry.pathology_registry.predict(batch_texts)

            # Generate HTML reports
            for report_index, (report, report_text) in enumerate(
                zip(batch_reports, batch_texts)
            ):
                report_html = generate_pathology_html_report(
                    report_text,
                    report_index,
                    batch_predictions,
                )
                report[REPORT_TEXT_COLUMN] = report_html

        elif predict_type == "mobility":
            # Get mobility annotations
            batch_entity_indexes = (
                model_registry.mobility_registry.extract_entity_indexes(batch_texts)
            )

            # Generate HTML reports
            for report_index, (report, report_text) in enumerate(
                zip(batch_reports, batch_texts)
            ):
                report_html = generate_mobility_html_report(
                    report_text,
                    report_index,
                    batch_entity_indexes,
                )
                report[REPORT_TEXT_COLUMN] = report_html

        logger.debug(f"Processed {len(batch_reports)}/{len(valid_reports)} reports")

    return reports
