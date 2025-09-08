import pytest
from botwenger.features import Features
from botwenger.config import INTERIM_DATA_DIR, INTERIM_DATA_FILENAME
from loguru import logger

data = Features.loading_preprocessed_data(f"{INTERIM_DATA_DIR}/{INTERIM_DATA_FILENAME}")
data_filled = Features.fill_fields_with_nas_for_basic_values(data)


def test_basic_features_filling():

    logger.info("Basic checks for filled values")

    data_filled = Features.fill_fields_with_nas_for_basic_values(data)

    assert data_filled["away_team_goals"].notna().all()
    assert data_filled["home_team_goals"].notna().all()  
    assert data_filled["sofascore_score"].notna().all()  
    assert data_filled["picas_as"].notna().all()  
    assert data_filled["status"].notna().all()  

def test_fill_market_price():

    logger.info("Testing filled market prices")
    data_filled_market = data_filled.groupby(['player', 'season'], group_keys=False).apply(Features.fill_market_price)
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
    

    
    
    



    

