import pytest
from botwenger.preprocessing import Preprocessing
from botwenger.config import RAW_DATA_DIR, RAW_DATA_FILENAME
from loguru import logger

data = Preprocessing.loading_raw_data(f"{RAW_DATA_DIR}/{RAW_DATA_FILENAME}")
data_fixed = data.groupby(["player", "season"], group_keys=False).apply(Preprocessing.fix_league_rounds)


def test_loading_raw_data():

    logger.info("DQ for is_player_home")
    assert data["is_player_home"].unique().size == 2
    assert data["is_player_home"].notna().all()

    logger.info("DQ for away_team_goals")
    assert data["away_team_goals"].min() == 0
    assert data["away_team_goals"].max() <= 13

    logger.info("DQ for home_team_goals")
    assert data["home_team_goals"].min() == 0
    assert data["home_team_goals"].max() <= 13

    logger.info("DQ for player_penalti_goals")
    assert data["player_penalti_goals"].min() == 0
    assert data["player_penalti_goals"].max() <= 5
    assert data["player_penalti_goals"].notna().all()

    logger.info("DQ for is_player_home")
    assert data["player_non_penalti_goals"].min() == 0
    assert data["player_non_penalti_goals"].max() <= 7   
    assert data["player_non_penalti_goals"].notna().all() 

    logger.info("DQ for player_assists")
    assert data["player_assists"].min() == 0
    assert data["player_assists"].max() <= 7
    assert data["player_assists"].notna().all() 

    logger.info("DQ for player_red_cards")
    assert data["player_red_card"].unique().size == 2
    assert data["player_red_card"].notna().all() 

    logger.info("DQ for player_second_yellow")
    assert data["player_second_yellow"].unique().size == 2
    assert data["player_second_yellow"].notna().all()     

    logger.info("DQ for home_team")
    assert data["home_team"].unique().size <= 50
    assert data["home_team"].notna().all()     

    logger.info("DQ for away_team")
    assert data["away_team"].unique().size <= 50
    assert data["away_team"].notna().all()     

    logger.info("DQ for league_round")
    assert data["league_round"].min() == 1
    assert data["league_round"].max() == 38 # Fixed later for null values

    logger.info("DQ for sofascore_score")
    assert data["sofascore_score"].min() >= 0
    assert data["sofascore_score"].max() <= 10

    logger.info("DQ for puntuacion_media_sofascore")
    assert data["puntuacion_media_sofascore_as"].min() >= -15
    assert data["puntuacion_media_sofascore_as"].max() <= 35

    logger.info("DQ for picas_as")
    assert data["picas_as"].unique().size >= 4 #Have to include SC somehow

    logger.info("DQ for player")
    assert data["player"].unique().size > 300
    assert data["player"].notna().all()     

    logger.info("DQ for player_price")
    assert data["player_price"].min() == 150000
    assert data["player_price"].max() <= 40000000

    logger.info("DQ for minutes_played")
    assert data["minutes_played"].min() == 0
    assert data["minutes_played"].max() == 90

    logger.info("DQ for player_position")
    assert data["player_position"].unique().size == 5 #1: GK, 2:DF 3: MC 4: ST 5: COACH (TO REMOVE)
    assert data["player_position"].notna().all()     

    logger.info("DQ for season")
    assert data["season"].unique().size == 8
    assert data["season"].notna().all()     

    logger.info("DQ for status")
    assert data["status"].unique().size >= 5

def test_player_season_ordered():

    logger.info("Checking raw data is ordered by player/season...")
    data_for_shape = data.copy()
    data_for_shape['player_season'] = list(zip(data_for_shape['player'], data_for_shape['season']))
    
    # Check that the index of each unique combination is contiguous
    for combo in data_for_shape['player_season'].unique():
        indices = data_for_shape.index[data_for_shape['player_season'] == combo].tolist()
        # Contiguous if max - min + 1 == number of indices
        assert max(indices) - min(indices) + 1 == len(indices)




def test_status_rounds_raw_data():
    logger.info("Checking there are only status values when no league round info "
    "and NAN status when league round info")

    assert len(data[(data["league_round"].isna()) & 
                       (data["status"].isna())]) == 0
    
    assert len(data[(data["league_round"].notna()) & 
                       (data["status"].notna())]) == 0


def test_fix_league_rounds():
    
    logger.info("Basic check new field")
    assert data_fixed.shape == (data.shape[0],) + (data.shape[1] + 1,) #Same rows, one more field
    
    logger.info("Check as many rounds as 40 (player moved teams)")
    assert data_fixed["fixed_round"].unique().size <= 40

    logger.info("Check the fixed rounds are unique and sequential for every player/season")
    grouped = data_fixed.groupby(["player", "season"])
    for _, group in grouped:
        fixed_rounds = group["fixed_round"].tolist()
        # Check all rounds are unique within the group
        assert len(fixed_rounds) == len(set(fixed_rounds))
        # Check rounds are sequential (difference between sorted values is always 1)
        diffs = [b - a for a, b in zip(fixed_rounds[:-1], fixed_rounds[1:])]
        assert all(diff == 1 for diff in diffs)

def test_fill_values_0():

    logger.info("Basic checks for filled values")
    data_filled = Preprocessing.fill_minutes_played_0(data_fixed)
    data_filled = Preprocessing.fill_puntuacion_media_0(data_filled)

    assert data_filled["minutes_played"].notna().all()  
    assert data_filled["puntuacion_media_sofascore_as"].notna().all()

    assert len(data_filled[(data_filled["player"]=="a-catena") & 
                       (data_filled["season"]==2025) & 
                       (data_filled["minutes_played"]==0)]) == 3
    
    assert len(data_filled[(data_filled["player"]=="zubeldia") & 
                       (data_filled["season"]==2019) & 
                       (data_filled["puntuacion_media_sofascore_as"]==0)]) == 6
    
    logger.info("Basic check shape after filling")
    assert data_filled.shape == data_fixed.shape
    
         
   


