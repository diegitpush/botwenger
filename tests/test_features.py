import pytest
from botwenger.features import Features
from botwenger.config import INTERIM_DATA_DIR, INTERIM_DATA_FILENAME
from loguru import logger
import numpy as np


data = Features.loading_preprocessed_data(f"{INTERIM_DATA_DIR}/{INTERIM_DATA_FILENAME}")

data_filled = Features.fill_fields_with_nas_for_basic_values(data)

data_filled_market = data_filled.groupby(['player', 'season'], group_keys=False).apply(Features.fill_market_price)

data_preselected_features = Features.prefilter_features_to_use(data_filled_market)

data_curated = Features.curate_and_simplify_features(data_preselected_features)

data_dummies = Features.create_dummies(data_curated)

data_rolling_past = data_dummies.copy()
data_rolling_past["puntuacion_media_roll_avg_8"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["puntuacion_media_sofascore_as"].transform(Features.past_rolling_avg_features)
data_rolling_past["red_card_roll_avg_8"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["player_red_card"].transform(Features.past_rolling_avg_features)

data_rolling_future = data_rolling_past.copy()
data_rolling_future["prediction_target_puntuacion_media_roll_avg_next_8"] = data_rolling_future.groupby(['player', 'season'], group_keys=False)["puntuacion_media_sofascore_as"].transform(Features.future_rolling_avg_target)

data_dropped_nans = Features.remove_nans_for_rolling_avgs(data_rolling_future)


def test_basic_features_filling():

    logger.info("Basic checks for filled values")

    assert data_filled["away_team_goals"].notna().all()
    assert data_filled["home_team_goals"].notna().all()  
    assert data_filled["sofascore_score"].notna().all()  
    assert data_filled["picas_as"].notna().all()  
    assert data_filled["status"].notna().all()  

def test_fill_market_price():

    logger.info("Testing filled market prices")

    assert data_filled_market["player_price"].notna().all()

    assert data_filled_market[(data_filled_market["player"]=="a-fernandez") & 
                       (data_filled_market["season"]==2023) & 
                       (data_filled_market["fixed_round"]==1)]["player_price"].iloc[0] == 410000
    
    assert data_filled_market[(data_filled_market["player"]=="a-fernandez") & 
                       (data_filled_market["season"]==2023) & 
                       (data_filled_market["fixed_round"]==5)]["player_price"].iloc[0] == 410000
    
    assert data_filled_market[(data_filled_market["player"]=="a-fernandez") & 
                       (data_filled_market["season"]==2023) & 
                       (data_filled_market["fixed_round"]==9)]["player_price"].iloc[0] == 410000
    
    assert data_filled_market[(data_filled_market["player"]=="a-fernandez") & 
                       (data_filled_market["season"]==2023) & 
                       (data_filled_market["fixed_round"]==11)]["player_price"].iloc[0] == 415000
    
    assert data_filled_market[(data_filled_market["player"]=="a-gorosabel") & 
                       (data_filled_market["season"]==2023) & 
                       (data_filled_market["fixed_round"]==20)]["player_price"].iloc[0] == 444286
    
    assert data_filled_market[(data_filled_market["player"]=="a-gorosabel") & 
                       (data_filled_market["season"]==2025) & 
                       (data_filled_market["fixed_round"]==38)]["player_price"].iloc[0] == 670000

def test_curate_and_simplify_features():

    logger.info("Testing cutating and simplifying features...")

    assert len(data_curated[data_curated["player_position"]==5]) == 0
    assert data_curated["status_mapped"].unique().size == 4
    assert "status" not in data_curated.columns
    

def test_create_dummies_for_status():

    logger.info("Testing the creation of dummies...")

    assert "status_mapped" not in data_dummies.columns
    assert "player_position" not in data_dummies.columns

    assert "status_mapped_ok" in data_dummies.columns
    assert "status_mapped_injured" in data_dummies.columns
    assert "status_mapped_sanctioned" in data_dummies.columns
    assert "status_mapped_doubt" in data_dummies.columns

    assert "player_position_1" in data_dummies.columns
    assert "player_position_2" in data_dummies.columns
    assert "player_position_3" in data_dummies.columns
    assert "player_position_4" in data_dummies.columns


def test_past_rolling_avgs():

    logger.info("Testing the past rolling averages...")
    
    assert data_rolling_past[(data_rolling_past["player"]=="a-catena") & 
                       (data_rolling_past["season"]==2025) & 
                       (data_rolling_past["fixed_round"]==38)]["puntuacion_media_roll_avg_8"].iloc[0] == 5.25
    
    assert data_rolling_past[(data_rolling_past["player"]=="a-catena") & 
                       (data_rolling_past["season"]==2025) & 
                       (data_rolling_past["fixed_round"]==6)]["puntuacion_media_roll_avg_8"].iloc[0] == 3

    assert data_rolling_past[(data_rolling_past["player"]=="a-catena") & 
                       (data_rolling_past["season"]==2025) & 
                       (data_rolling_past["fixed_round"]==26)]["puntuacion_media_roll_avg_8"].iloc[0] == 3.5
    
    assert np.isnan(data_rolling_past[(data_rolling_past["player"]=="a-catena") & 
                       (data_rolling_past["season"]==2025) & 
                       (data_rolling_past["fixed_round"]==2)]["puntuacion_media_roll_avg_8"].iloc[0])
    
    logger.info("Testing the NANs are the same for all past rolling averages...")

    assert len(data_rolling_past[(data_rolling_past["puntuacion_media_roll_avg_8"].isna())]) == len(data_rolling_past[(data_rolling_past["red_card_roll_avg_8"].isna())])
    

def test_future_rolling_avgs():

    logger.info("Testing the future rolling averages...")
    
    assert np.isnan(data_rolling_future[(data_rolling_future["player"]=="a-catena") & 
                       (data_rolling_future["season"]==2025) & 
                       (data_rolling_future["fixed_round"]==36)]["prediction_target_puntuacion_media_roll_avg_next_8"].iloc[0])
    
    assert data_rolling_future[(data_rolling_future["player"]=="a-catena") & 
                       (data_rolling_future["season"]==2025) & 
                       (data_rolling_future["fixed_round"]==33)]["prediction_target_puntuacion_media_roll_avg_next_8"].iloc[0] == 5.6

    assert data_rolling_future[(data_rolling_future["player"]=="a-catena") & 
                       (data_rolling_future["season"]==2025) & 
                       (data_rolling_future["fixed_round"]==2)]["prediction_target_puntuacion_media_roll_avg_next_8"].iloc[0] == 2.875
        

       
def test_remove_nans_for_rolling_avgs():

    logger.info("Testing the Nan dropping for rolling averages...")

    assert len(data_dropped_nans) < len(data_rolling_future)


def test_tbr():

    Features.main()