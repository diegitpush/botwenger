import pytest
from botwenger.preprocessing import Preprocessing
from botwenger.config import RAW_DATA_DIR

def test_loading_raw_data():
    data = Preprocessing.loading_raw_data(f"{RAW_DATA_DIR}/biwenger_players_history.tar.gz")
    print(data.info)
    pass

test_loading_raw_data()    
