from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RAW_DATA_FILENAME = "biwenger_players_history.tar.gz"
RAW_DATA_POINTS_TEAM = "liga_cumulative_points_2018_2025.csv"
INTERIM_DATA_FILENAME = "biwenger_players_history_preprocessed.csv"
PROCESSED_DATA_FILENAME_1 = "biwenger_features_processed_1.csv"
PROCESSED_DATA_FILENAME_8 = "biwenger_features_processed_8.csv"

MODELS_DIR = PROJ_ROOT / "models"
MODEL_FILENAME = "biwenger_[number_matches_to_predict]_match_points_predictor.json"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
