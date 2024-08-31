import os
from pathlib import Path
import logging
from dotenv import load_dotenv


load_dotenv()

MODE = os.getenv("MODE", "dev")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 4096))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

import logging
import os


def get_logger(name):
    # Determine the logging mode
    mode = MODE.lower()  # Default to "dev" if MODE is not set

    if mode == "debug":
        logging.basicConfig(level=logging.DEBUG)
    elif mode == "dev":
        logging.basicConfig(level=logging.INFO)
    else:
        Path("./log").mkdir(exist_ok=True)
        logging.basicConfig(filename="./log/nlp_app.log", level=logging.INFO)

    # Create a logger with the given name
    logger = logging.getLogger(name)
    return logger
