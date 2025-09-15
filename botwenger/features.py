from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
import time
from datetime import datetime
import re


from botwenger.config import INTERIM_DATA_DIR, INTERIM_DATA_FILENAME, RAW_DATA_DIR, RAW_DATA_POINTS_TEAM, PROCESSED_DATA_FILENAME_1, PROCESSED_DATA_FILENAME_8, PROCESSED_DATA_FILENAME_3

app = typer.Typer()

class Features: 

    preselected_features_training = ["player","season","puntuacion_media_sofascore_as","player_price","minutes_played",
                     "player_position","status","fixed_round", "is_player_home", "date", "home_team", "away_team"]
    
    preselected_features_inference = ["player","season","puntuacion_media_sofascore_as","player_price_now", "player_price_for_match","minutes_played",
                     "player_position","status","fixed_round", "is_player_home", "date", "home_team", "away_team", "roster", "status_info"]
    
    dummy_features = ["player_position","status_mapped"]

    status_map = {
    'discarded': 'ok',
    'doubt': 'doubt',
    'injured': 'injured',
    'ok': 'ok',
    'sanctioned': 'sanctioned',
    'unknown': 'ok',
    'warned': 'ok'
    }

    teams_map = {
    'athletic-bilbao': 'athletic',
    'atletico-madrid': 'atletico',
    'real-betis': 'betis',
    'celta-vigo': 'celta',
    'deportivo-la-coruna': 'deportivo',
    'huesca': 'sd-huesca'
    }

    final_selected_features_training = ["player_price", "fixed_round", "player_position_1",
                               "player_position_2","player_position_3","player_position_4",
                               "status_mapped_ok", "status_mapped_doubt",
                               "status_mapped_sanctioned","puntuacion_media_roll_avg_3",
                               "minutes_played_roll_avg_3",
                               "prediction_target_puntuacion_media_roll_avg",
                               "calculated_injury_severity", "player_team_strength",
                               "recent_price_change_1", "price_change_time_ratio" , "season", "player"] #season and players won't be features, season just used to split test/train and players for visibility
    
    final_selected_features_inference = ["player_price", "fixed_round", "player_position_1",
                               "player_position_2","player_position_3","player_position_4",
                               "status_mapped_ok", "status_mapped_doubt",
                               "status_mapped_sanctioned","puntuacion_media_roll_avg_3",
                               "minutes_played_roll_avg_3",
                               "calculated_injury_severity", "player_team_strength",
                               "recent_price_change_1", "price_change_time_ratio" , "season", "player",
                               "roster"] #roster won't be used in ML model, only on knapsack algorithm
    

    @app.command()
    @staticmethod    
    def features_training(output_dir: str = "data/processed", number_matches_to_predict: int = 1):

        logger.info("Starting feature engineering for training...")

        if number_matches_to_predict==1: 
            output_file = PROCESSED_DATA_FILENAME_1
        elif number_matches_to_predict==8:
            output_file = PROCESSED_DATA_FILENAME_8
        elif number_matches_to_predict==3:
            output_file = PROCESSED_DATA_FILENAME_3

        data = Features.loading_preprocessed_data(f"{INTERIM_DATA_DIR}/{INTERIM_DATA_FILENAME}")

        data_filled = Features.fill_fields_with_nas_for_basic_values(data)

        data_filled_market = data_filled.groupby(['player', 'season'], group_keys=False).apply(Features.fill_market_price, training = True)

        data_preselected_features = Features.prefilter_features_to_use(data_filled_market, training=True)

        data_curated = Features.curate_and_simplify_features(data_preselected_features)

        data_dummies = Features.create_dummies(data_curated, training=True)

        data_teams = Features.add_team_strength_feature(data_dummies)

        data_price_change = data_teams.copy()
        data_price_change["recent_price_change_1"] = data_price_change.groupby(['player', 'season'], group_keys=False)["player_price"].transform(Features.recent_price_change_training)

        data_matches_difference = data_price_change.copy()
        data_matches_difference["matches_date_difference"] = data_matches_difference.groupby(['player', 'season'], group_keys=False)["date"].transform(Features.matches_date_difference_training)

        data_price_change_ratio = Features.price_change_time_ratio(data_matches_difference)

        data_rolling_past = data_price_change_ratio.copy()
        data_rolling_past["puntuacion_media_roll_avg_3"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["puntuacion_media_sofascore_as"].transform(Features.past_rolling_avg_features)
        data_rolling_past["minutes_played_roll_avg_3"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["minutes_played"].transform(Features.past_rolling_avg_features)

        data_rolling_future = data_rolling_past.copy()
        data_rolling_future["prediction_target_puntuacion_media_roll_avg"] = data_rolling_future.groupby(['player', 'season'], group_keys=False)["puntuacion_media_sofascore_as"].transform(Features.future_rolling_avg_target_training, future_rows_number=number_matches_to_predict)

        data_injury_severity = data_rolling_future.copy()
        data_injury_severity["calculated_injury_severity"] = data_injury_severity.groupby(['player', 'season'], group_keys=False)["status_mapped_injured"].transform(Features.calculate_injury_severity_training)

        data_dropped_nans = Features.remove_nans_for_rolling_avgs(data_injury_severity)

        final_features = Features.final_features_select(data_dropped_nans, training= True)

        final_features.to_csv(f"{output_dir}/{output_file}", index=False)

        logger.success(f"Finished feature engineering for training. Saved in {output_dir}")

    @staticmethod    
    def features_inference(data: pd.DataFrame, number_matches_to_predict: int = 1) -> pd.DataFrame:

        data_filled = Features.fill_fields_with_nas_for_basic_values(data)

        data_filled_market = data_filled.groupby(['player', 'season'], group_keys=False).apply(Features.fill_market_price, training = False)

        data_last_matches = data_filled_market.copy()
        data_last_matches = data_filled_market.groupby(['player', 'season'], group_keys=False).apply(Features.filter_last_matches_inference)

        data_preselected_features = Features.prefilter_features_to_use(data_last_matches, training=False)

        data_curated = Features.curate_and_simplify_features(data_preselected_features)

        data_dummies = Features.create_dummies(data_curated, training = False)

        data_teams = Features.add_team_strength_feature(data_dummies)

        data_price_change = data_teams.copy()
        data_price_change = data_price_change.groupby(['player', 'season'], group_keys=False).apply(Features.recent_price_change_inference)

        data_matches_difference = data_price_change.copy()
        data_matches_difference["matches_date_difference"] = data_matches_difference.groupby(['player', 'season'], group_keys=False)["date"].transform(Features.matches_date_difference_inference)

        data_price_change_ratio = Features.price_change_time_ratio(data_matches_difference)

        data_rolling_past = data_price_change_ratio.copy()
        data_rolling_past["puntuacion_media_roll_avg_3"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["puntuacion_media_sofascore_as"].transform(Features.past_rolling_avg_features)
        data_rolling_past["minutes_played_roll_avg_3"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["minutes_played"].transform(Features.past_rolling_avg_features)

        data_injury_severity = data_rolling_past.copy()

        data_injury_severity.loc[data_injury_severity['status_mapped_injured'] == True, "calculated_injury_severity"] = data_injury_severity.loc[data_injury_severity['status_mapped_injured'] == True, "status_info"].apply(Features.calculate_injury_severity_inference)
        data_injury_severity.loc[data_injury_severity['status_mapped_injured'] == False, "calculated_injury_severity"] = 0

        final_features = Features.final_features_select(data_injury_severity, training=False)

        logger.success(f"Finished feature engineering for inference")

        final_features

    @staticmethod
    def loading_preprocessed_data(path: str) -> pd.DataFrame:
        logger.info("Loading preprocessed data...")
        data = pd.read_csv(path)
        logger.info("Loaded preprocessed data")
        return data


    @staticmethod    
    def fill_fields_with_nas_for_basic_values(data: pd.DataFrame) -> pd.DataFrame:

        logger.info("Filling Status with OK...(only NA when player played)")
        data["status"].fillna("ok", inplace=True)

        return data
    
    @staticmethod    
    def filter_last_matches_inference(group: pd.DataFrame) -> pd.DataFrame:

        logger.info("Filtering for only the latest matches...")
        group = group.nlargest(4, 'date').sort_values(by='date', ascending=True)
        return group
    
    @staticmethod    
    def fill_market_price(group: pd.DataFrame, training: bool) -> pd.DataFrame:

        logger.info("Filling NA marker prices with linear interpolation or repetiton...")

        if training:
            price_field = "player_price"
        elif not training:
            price_field = "player_price_for_match"    

        group = group.sort_values('fixed_round', ascending=True) #The order should already be like this

        # Interpolate linearly for internal missing values
        group[price_field] = group[price_field].interpolate(method='linear')

        # For any remaining NaNs at start or end, fill with nearest known value
        group[price_field] = group[price_field].ffill().bfill()

        if training:
            # Remaining with 150K, minimum value (for players that didn't play one minute all season)
            group[price_field].fillna(150000, inplace=True)
        elif not training:
            group[price_field].fillna(group["player_price_now"], inplace=True)
    
        group[price_field] = group[price_field].round().astype(int)

        return group
    
    @staticmethod
    def curate_and_simplify_features(data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Removing position = 5, as they are coaches...")
        data_filtered = data[data["player_position"].isin([1,2,3,4])]

        logger.info(f"Simplifying status...")
        data_filtered["status_mapped"] = data_filtered["status"].map(Features.status_map)

        data_filtered = data_filtered.drop(columns=['status'])

        return data_filtered

    
    @staticmethod    
    def prefilter_features_to_use(data: pd.DataFrame, training: bool) -> pd.DataFrame:
        logger.info(f"Preselecting features...")
        if training:
            data = data[Features.preselected_features_training] 
        elif not training:
            data = data[Features.preselected_features_inference]    
        return data
    

    @staticmethod    
    def create_dummies(data: pd.DataFrame, training: bool) -> pd.DataFrame:
        logger.info(f"Creating dummies for status field...")
        data = pd.get_dummies(data, columns=Features.dummy_features)

        if not training:
            required_cols = ["status_mapped_ok", "status_mapped_injured", "status_mapped_sanctioned", "status_mapped_doubt"]
            for col in required_cols:
                if col not in data.columns:
                    data[col] = False

        return data
    
    @staticmethod
    def past_rolling_avg_features(series: pd.DataFrame, past_rows_number: int = 3)-> pd.DataFrame:
        logger.info(f"Calculating rolling features for avg of last {past_rows_number} matches...")
        results = []
        n = len(series)
        for i in range(n):
            window = series.iloc[max(0, i-past_rows_number):i]  
            if len(window) >= 3: #if less than 3 previous matches, data won't be used for model
                results.append(window.mean()) 
            else:
                results.append(np.nan)
        return results
    
    @staticmethod
    def recent_price_change_training(series: pd.DataFrame, past_rows_number: int = 1)-> pd.DataFrame:
        logger.info(f"Calculating price change for last {past_rows_number} matches for training..")
        results = []
        n = len(series)
        for i in range(n):
            window = series.iloc[max(0, i-past_rows_number):i+1]
            results.append(window.iloc[-1] - window.iloc[0]) 
        return results
    
    @staticmethod
    def recent_price_change_inference(group: pd.DataFrame,)-> pd.DataFrame:
        logger.info(f"Calculating price change since now to last match for inference...")
        max_price = group.loc[group['date'].idxmax(), 'player_price_for_match']
        group['recent_price_change_1'] = group['player_price_now'].iloc[0] - max_price
        return group
    
    @staticmethod
    def matches_date_difference_training(series: pd.DataFrame)-> pd.DataFrame:
        logger.info(f"Calculating time passed since last match for training...")
        results = []
        n = len(series)
        for i in range(n):
            window = series.iloc[max(0, i-1):i+1]
            results.append(window.iloc[-1] - window.iloc[0])
        return results
    
    @staticmethod
    def matches_date_difference_inference(group: pd.DataFrame)-> pd.DataFrame:
        logger.info(f"Calculating time passed since last match for inference...")
        date_last_match = group.max()
        date_now = round(time.time())
        results = date_now - date_last_match
        return results
    
    @staticmethod
    def price_change_time_ratio(data: pd.DataFrame)-> pd.DataFrame:
        logger.info(f"Calculating price change/time ratio since last match...")
        data["price_change_time_ratio"] =np.where(data["matches_date_difference"] == 0, 0, data["recent_price_change_1"] / data["matches_date_difference"])
        return data
    
    
    @staticmethod
    def future_rolling_avg_target_training(series: pd.DataFrame, future_rows_number: int = 1)-> pd.DataFrame:
        logger.info(f"Calculating rolling future avg for target score of next {future_rows_number} matches...")

        if future_rows_number == 1:
            number_clipped_rows = 1
        elif future_rows_number == 8:
            number_clipped_rows = 3
        elif future_rows_number == 3:
            number_clipped_rows = 3
            
        results = []
        n = len(series)
        for i in range(n):
            window = series.iloc[i+1:i+1+future_rows_number]
            if len(window) >= number_clipped_rows: #if less than 3 future matches, data won't be used for model
                results.append(window.mean())
            else:
                results.append(np.nan) 

        return results
    
    @staticmethod
    def calculate_injury_severity_training(series: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Calculating injuries severity for training...")
        #Identify groups of consecutive 1s
        group = (series.ne(series.shift())
                 .cumsum())
        
        #Get run lengths only for 1-groups
        run_lengths = series.groupby(group).transform("sum")
        
        #Assign result: 0 stays 0, 1s get the remaining length in their run
        results = np.where(
            series == 1,
            run_lengths - series.groupby(group).cumcount(),
            0
        )

        return results
    

    @staticmethod
    def calculate_injury_severity_inference(status_info: str) -> float:
        logger.info(f"Parsing injury severity for inference...")

        dia_map = {
            "Principios": 5,
            "Mediados": 15,
            "Finales": 25
        }

        match = re.search(r"(Principios|Mediados|Finales) de (\w+)", status_info)

        if not match:
            match = re.search(r"(Vuelta indefinida)", status_info)
            if not match:
                raise Exception("Unable to match regex for injury severity")
            elif match:
                return 30

    
        periodo, mes_texto = match.groups()
    
        meses = {
                "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
            "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
        }

        mes_num = meses.get(mes_texto)

        if mes_num == None:
            raise Exception(f"Unable to match month for injury severity. Mes: {mes_texto}")

        dia = dia_map.get(periodo)

        if dia == None:
            raise Exception(f"Unable to match day for injury severity. Dia: {periodo}")

        hoy = datetime.today()
        year = hoy.year
    
        if mes_num < hoy.month or (mes_num == hoy.month and dia < hoy.day):
            year += 1
            
        matches_injured_left = (datetime(year, mes_num, dia) - datetime.today()).days/7

        matches_injured_left_int = round(matches_injured_left)

        return matches_injured_left_int

    
    @staticmethod
    def add_team_strength_feature(data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Mapping players to team strength...")

        logger.info(f"Loading team points historical info...")
        points_team = pd.read_csv(f"{RAW_DATA_DIR}/{RAW_DATA_POINTS_TEAM}", header = None, names = ["team", "points_per_season"])
        points_team = points_team.groupby("team")["points_per_season"].sum().sort_values(ascending=0).reset_index()
        points_team["team"] = points_team["team"].str.lower().str.replace(' ', '-', regex=False)
        points_team["team"] = points_team["team"].replace(Features.teams_map)

        logger.info(f"Adding new team strength feature...")

        data['team'] = np.where(data['is_player_home'], data['home_team'], data['away_team'])

        data_new_feature = data.merge(points_team, on='team', how='left')

        data_new_feature = data_new_feature.rename(columns={"points_per_season": "player_team_strength"})

        return data_new_feature

    
    @staticmethod
    def remove_nans_for_rolling_avgs(data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Removing NANs for target rolling avg...")

        data = data.dropna(subset=["prediction_target_puntuacion_media_roll_avg"])

        return data
    
    @staticmethod    
    def final_features_select(data: pd.DataFrame, training: bool) -> pd.DataFrame:
        logger.info(f"Final selection of features...")
        if training:

            data = data[Features.final_selected_features_training]

        elif not training:

            data["player_price"] = data["player_price_now"]
            data = data[Features.final_selected_features_inference]

        return data

if __name__ == "__main__":
    app()
