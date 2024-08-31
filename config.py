import os
from pathlib import Path
import sys

from loguru import logger
from dotenv import load_dotenv


load_dotenv()


# Environment variables
DEBUG = os.getenv("DEBUG", False)
MODE = os.getenv("MODE", "development")
# Maximum length of input sequence to truncate
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 4096))
# Batch size for prediction
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

# Configure Loguru based on MODE
if MODE == "production":
    log_file = Path("./log/nlp_app.log")
    log_file.parent.mkdir(exist_ok=True)
    logger.add(log_file, level="INFO", rotation="10 MB")
else:
    logger.remove()  # Remove default stderr logging
    logger.add(sys.stderr, level="DEBUG" if DEBUG else "INFO")

logger.info(f"Mode: {MODE}. Debug: {DEBUG}")
