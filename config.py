import os
from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
MODE = os.getenv("MODE", "dev")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 4096))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

if MODE == "debug":
    logging.basicConfig(level=logging.DEBUG)
elif MODE == "dev":
    logging.basicConfig(level=logging.INFO)
else:
    Path("./log").mkdir(exist_ok=True)
    logging.basicConfig(filename="./log/nlp_app.log", level=logging.INFO)

logger = logging.getLogger(__name__)