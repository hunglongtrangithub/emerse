from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader

from .config import logger, BATCH_SIZE, REPORT_TEXT_COLUMN
from .models import ModelRegistry, PathologyPrediction

# Load the Jinja2 environment
template_dir = "./templates"
env = Environment(loader=FileSystemLoader(template_dir))
template = env.get_template("report_template.html")


def generate_html_report(
    report_text: str, batch_predictions: list[PathologyPrediction], index: int
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

    # Render the template with the report text and predictions data
    html_content = template.render(report_text=report_text, predictions=predictions)
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

        batch_predictions = model_registry.predict(batch_texts)

        for i, (report, report_text) in enumerate(zip(batch_reports, batch_texts)):
            report_html = generate_html_report(report_text, batch_predictions, i)
            # Add the HTML table to the report json object
            report[REPORT_TEXT_COLUMN] = report_html

        logger.debug(f"Predicted {len(batch_reports)}/{len(valid_reports)} reports")

    return reports
