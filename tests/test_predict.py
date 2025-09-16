import pytest
from botwenger.modeling.predict import Predict
from botwenger.config import INTERIM_DATA_DIR, INTERIM_DATA_FILENAME
from loguru import logger
import numpy as np


def test_get_and_preprocess_daily_data_api():

    pre_data = Predict.daily_recommended_changes()




