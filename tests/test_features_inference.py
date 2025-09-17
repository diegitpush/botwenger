import pytest
from botwenger.features import Features
from botwenger.config import INTERIM_DATA_DIR, INTERIM_DATA_FILENAME, TEST_DATA_INFERENCE, TEST_DATA_DIR
from loguru import logger
import numpy as np
from botwenger.modeling.predict import Predict
import pandas as pd

data = pd.read_csv(f"{TEST_DATA_DIR}/{TEST_DATA_INFERENCE}")

data_filled = Features.fill_fields_with_nas_for_basic_values(data)

data_filled_market = data_filled.groupby(['player'], group_keys=False).apply(Features.fill_market_price, training =  False)

data_last_matches = data_filled_market.copy()
data_last_matches = data_filled_market.groupby(['player'], group_keys=False).apply(Features.filter_last_matches_inference)

data_preselected_features = Features.prefilter_features_to_use(data_last_matches, training=False)

data_curated = Features.curate_and_simplify_features(data_preselected_features)

data_dummies = Features.create_dummies(data_curated, training = False)

data_teams = Features.add_team_strength_feature(data_dummies)

data_price_change = data_teams.copy()
data_price_change = data_price_change.groupby(['player'], group_keys=False).apply(Features.recent_price_change_inference)

data_matches_difference = data_price_change.copy()
data_matches_difference["matches_date_difference"] = data_matches_difference.groupby(['player'], group_keys=False)["date"].transform(Features.matches_date_difference_inference)

data_price_change_ratio = Features.price_change_time_ratio(data_matches_difference)

data_rolling_past = data_price_change_ratio.copy()
data_rolling_past["puntuacion_media_roll_avg_3"] = data_rolling_past.groupby(['player'], group_keys=False)["puntuacion_media_sofascore_as"].transform(Features.past_rolling_avg_features, training = False)
data_rolling_past["minutes_played_roll_avg_3"] = data_rolling_past.groupby(['player'], group_keys=False)["minutes_played"].transform(Features.past_rolling_avg_features, training = False)

data_injury_severity = data_rolling_past.copy()

data_injury_severity.loc[data_injury_severity['status_mapped_injured'] == True, "calculated_injury_severity"] = data_injury_severity.loc[data_injury_severity['status_mapped_injured'] == True, "status_info"].apply(Features.calculate_injury_severity_inference)
data_injury_severity.loc[data_injury_severity['status_mapped_injured'] == False, "calculated_injury_severity"] = 0
data_injury_severity["calculated_injury_severity"] = data_injury_severity["calculated_injury_severity"].astype(int)

data_last_match = Features.get_only_last_match_inference(data_injury_severity)

data_final_features = Features.final_features_select(data_last_match, training=False)

def test_basic_features_filling():

    logger.info("Basic checks for filled values")

    assert data_filled["status"].notna().all()  

def test_fill_market_price():

    logger.info("Testing filled market prices")

    assert data_filled_market["player_price_for_match"].notna().all()

    assert data_filled_market[(data_filled_market["player"]=="carlos-dominguez") & 
                       (data_filled_market["fixed_round"]==1)]["player_price_for_match"].iloc[0] == 180000
    
    assert data_filled_market[(data_filled_market["player"]=="carlos-dominguez") & 
                       (data_filled_market["fixed_round"]==3)]["player_price_for_match"].iloc[0] == 205000
    

def test_filter_last_matches_inference():

    logger.info("Basic checks for filtering of last matches for inference")

    assert (data_last_matches.groupby("player").size() <= 4).all()

def test_curate_and_simplify_features():

    logger.info("Testing cutating and simplifying features...")

    assert len(data_curated[data_curated["player_position"]==5]) == 0
    assert data_curated["status_mapped"].unique().size <= 4
    assert "status" not in data_curated.columns
    

def test_add_team_strength_feature():

    logger.info("Testing the addition of team strength feature...")

    assert data_teams["player_team_strength"].unique().size <= 29
    assert data_teams["player_team_strength"].notna().all()

    assert data_teams[(data_teams["player"]=="tchouameni")]["player_team_strength"].unique().size == 1
    
    assert data_teams[(data_teams["player"]=="tchouameni")]["player_team_strength"].unique().item(0) == 658
    

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


def test_recent_price_change():

    logger.info("Testing recent price change calculation...")
    
    assert data_price_change[(data_price_change["player"]=="alvaro-garcia") & 
                       (data_price_change["fixed_round"]==4)]["recent_price_change_1"].iloc[0] == 120000
    
    
def test_matches_date_difference():

    logger.info("Testing matches date difference change calculation...") 
    
    assert data_matches_difference["matches_date_difference"].min() >= 0


def test_price_change_time_ratio():

    logger.info("Testing price change/time ratio calculation...")
    
    assert data_price_change_ratio[(data_price_change_ratio["player"]=="alvaro-garcia") & 
                       (data_price_change_ratio["fixed_round"]==4)]["price_change_time_ratio"].iloc[0] <= 0
    
    

def test_past_rolling_avgs():

    logger.info("Testing the past rolling averages...")
    
    assert data_rolling_past[(data_rolling_past["player"]=="alvaro-garcia") & 
                       (data_rolling_past["fixed_round"]==4)]["puntuacion_media_roll_avg_3"].iloc[0] == 4
    
    assert np.isnan(data_rolling_past[(data_rolling_past["player"]=="alvaro-garcia") & 
                       (data_rolling_past["fixed_round"]==1)]["puntuacion_media_roll_avg_3"].iloc[0])
    
    logger.info("Testing the NANs are the same for all past rolling averages...")

    assert len(data_rolling_past[(data_rolling_past["puntuacion_media_roll_avg_3"].isna())]) == len(data_rolling_past[(data_rolling_past["minutes_played_roll_avg_3"].isna())])
        
def test_calculate_injury_severity():

    logger.info("Testing the calculation of injury severity...")

    assert data_injury_severity[(data_injury_severity["player"]=="alvaro-garcia") & 
                       (data_injury_severity["fixed_round"]==1)]["calculated_injury_severity"].iloc[0] == 40
    
    assert data_injury_severity[(data_injury_severity["player"]=="alvaro-garcia") &  
                       (data_injury_severity["fixed_round"]==3)]["calculated_injury_severity"].iloc[0] == 0
 
       
       