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
    entities = {
        entity_name: sorted(
            [
                {"start": start, "end": end}
                for start, end in entity_indexes[index].entity_indexes
            ],
            key=lambda x: x["start"],
        )
        for entity_name, entity_indexes in batch_entity_indexes.items()
    }

    html_content = mobility_template.render(
        report_text=report_text,
        entities=entities,
    )
    return html_content


def generate_mobility_document_with_entities(
    report_text: str,
    index: int,
    batch_entity_indexes: dict[str, list[MobilityPrediction]],
) -> str:
    """
    Generate a file string for the mobility document with entities compliant with the EMERSE NLP format.

    Args:
        report_text: The text of the report.
        index: The index of the current report in the batch.
        batch_entity_indexes: A dictionary mapping entity types to their respective MobilityPrediction lists.

    Returns:
        A formatted string compliant with the EMERSE NLP pipeline.
    """
    # Prepare the entities section
    entities = []
    for entity_name, entity_indexes in batch_entity_indexes.items():
        entities.extend(
            [
                {"type": entity_name, "start": start, "end": end}
                for start, end in entity_indexes[index].entity_indexes
            ]
        )

    # Header row (hardcoded example values for newline and space counts, and a separator)
    separator = "THIS_IS_A_SEPARATOR"
    header_row = f"RU1FUlNFX0g=1|5|{separator}"

    # NLP artifacts section
    cui_code = 2371377
    artifact_lines = []
    for entity in entities:
        start_offset = entity["start"]
        end_offset = entity["end"]
        match entity["type"]:
            case "Mobility":
                artifact_id = f"CUI_{cui_code}"
            case "Action":
                artifact_id = f"CUI_{cui_code}_a"
            case "Quantification":
                artifact_id = f"CUI_{cui_code}_q"
            case "Assistance":
                artifact_id = f"CUI_{cui_code}_s"
            case _:
                raise ValueError(f"Invalid entity type: {entity['type']}")
        artifact_type = "R"  # All entities are "R" for "Entity"
        artifact_lines.append(
            f"{start_offset}\t{end_offset}\t{artifact_id}\t{artifact_type}"
        )

    # Combine the artifact lines into a single string
    artifacts_section = "\n".join(artifact_lines)

    # Original document section
    document_section = report_text.strip()
    print(f"Document section: {document_section[:100]}...")

    # Combine all sections into the final document
    emerse_document = (
        (f"{header_row}\n{artifacts_section}\n{separator}\n{document_section}")
        if artifacts_section
        else document_section  # TODO: Check if this is correct. Do we need to add the header row if there are no artifacts?
    )
    return emerse_document


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
    Mutates the input reports by adding HTML content.

    Args:
        reports: List of report dictionaries
        model_registry: Model registry instance
        predict_type: Type of prediction to perform ("pathology" or "mobility")

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
            if model_registry.pathology_registry is None:
                logger.error("Pathology model not found in the model registry.")
                return reports
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
            if model_registry.mobility_registry is None:
                logger.error("Mobility model not found in the model registry.")
                return reports
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

                report_doc = generate_mobility_document_with_entities(
                    report_text,
                    report_index,
                    batch_entity_indexes,
                )
                # TODO: Consult with the EMERSE team to determine the correct key for the document
                report["PRT_DOC"] = report_doc

        logger.debug(f"Processed {len(batch_reports)}/{len(valid_reports)} reports")

    return reports
