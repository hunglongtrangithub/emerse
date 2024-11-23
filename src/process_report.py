from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader

from .config import logger, BATCH_SIZE, REPORT_TEXT_COLUMN
from .models import ModelRegistry, PathologyPrediction, MobilityPrediction

# Load the Jinja2 environment
template_dir = "./templates"
env = Environment(loader=FileSystemLoader(template_dir))
template = env.get_template("report_template.html")

COLOR_MAP = {
    "Action": "#F7DC6F",  # Pale Yellow
    "Assistant": "#BB8FCE",  # Purple
    "Mobility": "#F0B27A",  # Light Orange
    "Quantification": "#AED6F1",  # Light Cyan
}


def annotate_report_text(report_text, entity_indexes):
    """
    Annotates the report text with HTML tags and distinct colors based on entity types.

    :param report_text: The original report text.
    :param entity_indexes: A dictionary where keys are entity names and values are lists of
                           tuples indicating the start and end positions of the entities.
    :return: Annotated report text with HTML span tags for highlighting.
    """
    # Flatten and sort all entities by start position
    all_entities = []
    for entity_name, indexes in entity_indexes.items():
        for start, end in indexes:
            all_entities.append((start, end, entity_name))
    all_entities.sort(key=lambda x: x[0])

    # Annotate the text
    annotated_text = ""
    current_pos = 0
    for start, end, entity_name in all_entities:
        if current_pos < start:
            annotated_text += report_text[
                current_pos:start
            ]  # Add text before the entity
        color = COLOR_MAP.get(
            entity_name, "#FFFFFF"
        )  # Default to white if entity is not in COLOR_MAP
        annotated_text += (
            f'<span style="background-color: {color}; border-radius: 3px; padding: 2px;" title="{entity_name}">'
            f"{report_text[start:end]} <b>[{entity_name}]</b>"
            f"</span>"
        )
        current_pos = end
    if current_pos < len(report_text):
        annotated_text += report_text[
            current_pos:
        ]  # Add remaining text after the last entity
    return annotated_text


def generate_html_report(
    report_text: str,
    index: int,
    batch_predictions: list[PathologyPrediction],
    batch_entity_indexes: dict[str, list[MobilityPrediction]],
):
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
            "field_name": prediction.field_name,
            "field": prediction.field,
            "value": prediction.value[index],
            "max_prob": prediction.max_prob[index],
        }
        for prediction in batch_predictions
    ]
    entity_indexes = {
        entity_name: [
            (index_pair[0], index_pair[1])
            for index_pair in entity_indexes[index].entity_indexes
        ]
        for entity_name, entity_indexes in batch_entity_indexes.items()
    }

    # Annotate the report text
    annotated_report_text = annotate_report_text(report_text, entity_indexes)

    # Render the template with the annotated report text and predictions data
    html_content = template.render(
        annotated_report_text=annotated_report_text,
        predictions=predictions,
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


def get_reports_with_table(
    reports: list[dict[str, str]], model_registry: ModelRegistry
) -> list[dict[str, str]]:
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

        batch_predictions = model_registry.pathology_registry.predict(batch_texts)
        batch_entity_indexes = model_registry.mobility_registry.extract_entity_indexes(
            batch_texts
        )
        for i, (report, report_text) in enumerate(zip(batch_reports, batch_texts)):
            report_html = generate_html_report(
                report_text, i, batch_predictions, batch_entity_indexes
            )
            # Add the HTML table to the report json object
            report[REPORT_TEXT_COLUMN] = report_html

        logger.debug(f"Predicted {len(batch_reports)}/{len(valid_reports)} reports")

    return reports
