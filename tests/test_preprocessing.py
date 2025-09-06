import pytest
from botwenger.preprocessing import Preprocessing
from botwenger.config import RAW_DATA_DIR
from loguru import logger

data = Preprocessing.loading_raw_data(f"{RAW_DATA_DIR}/biwenger_players_history.tar.gz")

def test_loading_raw_data():

    logger.info("DQ for is_player_home")
    assert data["is_player_home"].unique().size == 2

    logger.info("DQ for away_team_goals")
    assert data["away_team_goals"].min() == 0
    assert data["away_team_goals"].max() <= 13

    logger.info("DQ for home_team_goals")
    assert data["home_team_goals"].min() == 0
    assert data["home_team_goals"].max() <= 13

    logger.info("DQ for player_penalti_goals")
    assert data["player_penalti_goals"].min() == 0
    assert data["player_penalti_goals"].max() <= 5

    logger.info("DQ for is_player_home")
    assert data["player_non_penalti_goals"].min() == 0
    assert data["player_non_penalti_goals"].max() <= 7    

    logger.info("DQ for player_assists")
    assert data["player_assists"].min() == 0
    assert data["player_assists"].max() <= 7

    logger.info("DQ for player_red_cards")
    assert data["player_red_card"].unique().size == 2

    logger.info("DQ for player_second_yellow")
    assert data["player_second_yellow"].unique().size == 2

    logger.info("DQ for home_team")
    assert data["home_team"].unique().size <= 50

    logger.info("DQ for away_team")
    assert data["away_team"].unique().size <= 50

    logger.info("DQ for league_round")
    assert data["league_round"].min() == 1
    assert data["league_round"].max() == 38

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

    logger.info("DQ for player_price")
    assert data["player_price"].min() == 150000
    assert data["player_price"].max() <= 40000000

    logger.info("DQ for minutes_played")
    assert data["minutes_played"].min() == 0
    assert data["minutes_played"].max() == 90

    logger.info("DQ for player_position")
    assert data["player_position"].unique().size == 5 #1: GK, 2:DF 3: MC 4: ST 5: COACH (TO REMOVE)

    logger.info("DQ for season")
    assert data["season"].unique().size == 8

    logger.info("DQ for status")
    assert data["status"].unique().size >= 5

def test_fix_league_rounds():
    data_fixed = data.groupby(["player", "season"], group_keys=False).apply(Preprocessing.fix_league_rounds)
    
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

