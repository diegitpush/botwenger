from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np

from botwenger.config import INTERIM_DATA_DIR, INTERIM_DATA_FILENAME

app = typer.Typer()

class Features: 

    preselected_features = ["player","season","player_red_card","player_non_penalti_goals","player_penalti_goals",
                     "puntuacion_media_sofascore_as","player_price","minutes_played",
                     "player_position","status","player_assists","player_second_yellow",
                     "fixed_round"] #player to be removed later, used to calculate rolling features
    
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

    final_selected_features = ["player_price", "fixed_round", "player_position_1",
                               "player_position_2","player_position_3","player_position_4",
                               "status_mapped_ok", "status_mapped_doubt",
                               "status_mapped_sanctioned","puntuacion_media_roll_avg_8",
                               "red_card_roll_avg_8","second_yellow_roll_avg_8",
                               "penalti_goals_roll_avg_8","non_penalti_goals_media_roll_avg_8",
                               "assists_roll_avg_8","minutes_played_roll_avg_8",
                               "prediction_target_puntuacion_media_roll_avg_next_8",
                               "calculated_injury_severity", "season"] #season won't be a feature, just used to split test/train

    @app.command()
    @staticmethod    
    def main(output_dir: str = "data/processed"):

        logger.info("Starting feature engineering...")

        data = Features.loading_preprocessed_data(f"{INTERIM_DATA_DIR}/{INTERIM_DATA_FILENAME}")

        data_filled = Features.fill_fields_with_nas_for_basic_values(data)

        data_filled_market = data_filled.groupby(['player', 'season'], group_keys=False).apply(Features.fill_market_price)

        data_preselected_features = Features.prefilter_features_to_use(data_filled_market)

        data_curated = Features.curate_and_simplify_features(data_preselected_features)

        data_dummies = Features.create_dummies(data_curated)

        data_rolling_past = data_dummies.copy()
        data_rolling_past["puntuacion_media_roll_avg_8"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["puntuacion_media_sofascore_as"].transform(Features.past_rolling_avg_features)
        data_rolling_past["red_card_roll_avg_8"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["player_red_card"].transform(Features.past_rolling_avg_features)
        data_rolling_past["second_yellow_roll_avg_8"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["player_second_yellow"].transform(Features.past_rolling_avg_features)
        data_rolling_past["penalti_goals_roll_avg_8"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["player_penalti_goals"].transform(Features.past_rolling_avg_features)
        data_rolling_past["non_penalti_goals_media_roll_avg_8"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["player_non_penalti_goals"].transform(Features.past_rolling_avg_features)
        data_rolling_past["assists_roll_avg_8"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["player_assists"].transform(Features.past_rolling_avg_features)
        data_rolling_past["minutes_played_roll_avg_8"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["minutes_played"].transform(Features.past_rolling_avg_features)

        data_rolling_future = data_rolling_past.copy()
        data_rolling_future["prediction_target_puntuacion_media_roll_avg_next_8"] = data_rolling_future.groupby(['player', 'season'], group_keys=False)["puntuacion_media_sofascore_as"].transform(Features.future_rolling_avg_target)

        data_injury_severity = data_rolling_future.copy()
        data_injury_severity["calculated_injury_severity"] = data_injury_severity.groupby(['player', 'season'], group_keys=False)["status_mapped_injured"].transform(Features.calculate_injury_severity)

        data_dropped_nans = Features.remove_nans_for_rolling_avgs(data_injury_severity)

        final_features = Features.final_features_select(data_dropped_nans)

        final_features.to_csv(f"{output_dir}/biwenger_features_processed_test.csv", index=False)

        logger.success(f"Finished feature engineering. Saved in {output_dir}")

    @staticmethod
    def loading_preprocessed_data(path: str) -> pd.DataFrame:
        logger.info("Loading preprocessed data...")
        data = pd.read_csv(path)
        logger.info("Loaded preprocessed data")
        return data


    @staticmethod    
    def fill_fields_with_nas_for_basic_values(data: pd.DataFrame) -> pd.DataFrame:

        logger.info("Filling NA team goals with 1...(most common)")
        data["away_team_goals"].fillna(1, inplace=True)
        data["home_team_goals"].fillna(1, inplace=True)

        logger.info("Filling NA SofaScore score with 6.0...(gives 0 points)")
        data["sofascore_score"].fillna(6.0, inplace=True)

        logger.info("Filling NA Picas AS with SC...")
        data["picas_as"].fillna("SC", inplace=True)  

        logger.info("Filling Status with OK...(only NA when player played)")
        data["status"].fillna("ok", inplace=True)

        return data
    
    @staticmethod    
    def fill_market_price(group: pd.DataFrame) -> pd.DataFrame:

        logger.info("Filling NA marker prices with linear interpolation or repetiton...")

        group = group.sort_values('fixed_round', ascending=True) #The order should already be like this

        # Interpolate linearly for internal missing values
        group['player_price'] = group['player_price'].interpolate(method='linear')

        # For any remaining NaNs at start or end, fill with nearest known value
        group['player_price'] = group['player_price'].ffill().bfill()

        # Remaining with 150K, minimum value (for players that didn't play one minute all season)
        group['player_price'].fillna(150000, inplace=True)

        group['player_price'] = group['player_price'].round().astype(int)

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
    def prefilter_features_to_use(data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Preselecting features in {Features.preselected_features}...")
        data = data[Features.preselected_features] 
        return data
    

    @staticmethod    
    def create_dummies(data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Creating dummies for status field...")
        data = pd.get_dummies(data, columns=Features.dummy_features)
        return data
    
    @staticmethod
    def past_rolling_avg_features(series: pd.DataFrame, past_rows_number: int = 8)-> pd.DataFrame:
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
    def future_rolling_avg_target(series: pd.DataFrame, future_rows_number: int = 8)-> pd.DataFrame:
        logger.info(f"Calculating rolling future avg for target score of next {future_rows_number} matches...")
        results = []
        n = len(series)
        for i in range(n):
            window = series.iloc[i+1:i+1+future_rows_number]
            if len(window) >= 3: #if less than 3 future matches, data won't be used for model
                results.append(window.mean())
            else:
                results.append(np.nan) 

        return results
    
    @staticmethod
    def calculate_injury_severity(series: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Calculating injuries severity...")
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
    def remove_nans_for_rolling_avgs(data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Removing NANs for rolling avgs...")

        # We use 3 to 8 previous matches, if less available we don't use that data. Same for the next 3 to 8 matches for target, if less we don't use it
        # We use puntuacion_media_roll_avg_8 as a proxy for all past roll avg fields, as they should share the same nans, and prediction_target_puntuacion_media_roll_avg_next_8 for future roll avgs nans

        data = data.dropna(subset=["puntuacion_media_roll_avg_8", "prediction_target_puntuacion_media_roll_avg_next_8"])

        return data
    
    @staticmethod    
    def final_features_select(data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Final selection of features in {Features.final_selected_features}...")
        data = data[Features.final_selected_features] 
        return data

if __name__ == "__main__":
    app()
