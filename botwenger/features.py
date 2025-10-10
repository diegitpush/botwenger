from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
import time
from datetime import datetime
import re


from botwenger.config import INTERIM_DATA_DIR, INTERIM_DATA_FILENAME, RAW_DATA_DIR, RAW_DATA_TEAM_RANK, PROCESSED_DATA_FILENAME_1, PROCESSED_DATA_FILENAME_8, PROCESSED_DATA_FILENAME_3

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
    def features_training(output_dir: str = "data/processed", number_matches_to_predict: int = 1, is_test: bool = False):
        """
        Performs feature engineering for training a predictive model on football player data.

        This function processes preprocessed data through several feature engineering steps,
        including filling missing values, market price adjustments, feature selection, dummy variable creation,
        team strength calculation, price change computation, rolling averages, injury severity calculation,
        and final feature selection. The processed features are saved to a CSV file or returned as intermediate
        DataFrames for testing purposes.

        Args:
            output_dir (str, optional): Directory to save the processed features CSV file. Defaults to "data/processed".
            number_matches_to_predict (int, optional): Number of future matches to predict (affects rolling averages and targets).
                Supported values are 1, 3, or 8. Defaults to 1.
            is_test (bool, optional): If True, returns a dictionary of intermediate DataFrames for testing and debugging.
                If False, saves the final features to a CSV file. Defaults to False.

        Returns:
            dict (optional): If is_test is True, returns a dictionary containing intermediate DataFrames at each step of
                the feature engineering pipeline. Otherwise, returns None.

        Raises:
            ValueError: If number_matches_to_predict is not one of the supported values (1, 3, or 8).

        Example:
            features_training(output_dir="data/processed", number_matches_to_predict=3, is_test=False)
        """

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
        data_rolling_past["puntuacion_media_roll_avg_3"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["puntuacion_media_sofascore_as"].transform(Features.past_rolling_avg_features, training = True)
        data_rolling_past["minutes_played_roll_avg_3"] = data_rolling_past.groupby(['player', 'season'], group_keys=False)["minutes_played"].transform(Features.past_rolling_avg_features, training = True)

        data_rolling_future = data_rolling_past.copy()
        data_rolling_future["prediction_target_puntuacion_media_roll_avg"] = data_rolling_future.groupby(['player', 'season'], group_keys=False)["puntuacion_media_sofascore_as"].transform(Features.future_rolling_avg_target_training, future_rows_number=number_matches_to_predict)

        data_injury_severity = data_rolling_future.copy()
        data_injury_severity["calculated_injury_severity"] = data_injury_severity.groupby(['player', 'season'], group_keys=False)["status_mapped_injured"].transform(Features.calculate_injury_severity_training)

        data_dropped_nans = Features.remove_nans_for_rolling_avgs(data_injury_severity)

        final_features = Features.final_features_select(data_dropped_nans, training= True)

        if is_test:
            return {
                "data": data,
                "data_filled": data_filled,
                "data_filled_market": data_filled_market,
                "data_preselected_features": data_preselected_features,
                "data_curated": data_curated,
                "data_dummies": data_dummies,
                "data_teams": data_teams,
                "data_price_change": data_price_change,
                "data_matches_difference": data_matches_difference,
                "data_price_change_ratio": data_price_change_ratio,
                "data_rolling_past": data_rolling_past,
                "data_rolling_future": data_rolling_future,
                "data_injury_severity": data_injury_severity,
                "data_dropped_nans": data_dropped_nans,
                "final_features": final_features
            }
        else:
            final_features.to_csv(f"{output_dir}/{output_file}", index=False)
            logger.success(f"Finished feature engineering for training. Saved in {output_dir}")

    @staticmethod    
    def features_inference(data: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        """
        Processes input player match data through a series of feature engineering steps for inference.
        The function applies multiple transformations and feature creation steps to the input DataFrame,
        including filling missing values, market price adjustments, filtering last matches, feature selection,
        dummy variable creation, team strength calculation, recent price change computation, match date difference,
        price change ratio, rolling averages for past performance, injury severity calculation, and final feature selection.
        If `is_test` is True, returns a dictionary containing intermediate DataFrames at each step for debugging or analysis.
        Otherwise, returns the final processed DataFrame ready for inference.
        Args:
            data (pd.DataFrame): Input DataFrame containing player match data.
            is_test (bool, optional): If True, returns intermediate results. Defaults to False.
        Returns:
            pd.DataFrame or dict: Final processed DataFrame for inference, or a dictionary of intermediate DataFrames if `is_test` is True.
        """


        data_filled = Features.fill_fields_with_nas_for_basic_values(data)

        data_filled_market = data_filled.groupby(['player'], group_keys=False).apply(Features.fill_market_price, training = False)

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

        if is_test:
            return {
                "data": data,
                "data_filled": data_filled,
                "data_filled_market": data_filled_market,
                "data_last_matches": data_last_matches,
                "data_preselected_features": data_preselected_features,
                "data_curated": data_curated,
                "data_dummies": data_dummies,
                "data_teams": data_teams,
                "data_price_change": data_price_change,
                "data_matches_difference": data_matches_difference,
                "data_price_change_ratio": data_price_change_ratio,
                "data_rolling_past": data_rolling_past,
                "data_injury_severity": data_injury_severity,
                "data_last_match": data_last_match,
                "data_final_features": data_final_features
            }
        
        else:
            return data_final_features

    @staticmethod
    def loading_preprocessed_data(path: str) -> pd.DataFrame:
        """
        Loads preprocessed data from a CSV file into a pandas DataFrame.
        Args:
            path (str): The file path to the CSV file containing preprocessed data.
        Returns:
            pd.DataFrame: The loaded preprocessed data as a pandas DataFrame.
        """

        logger.info("Loading preprocessed data...")
        data = pd.read_csv(path)
        logger.info("Loaded preprocessed data")
        return data


    @staticmethod    
    def fill_fields_with_nas_for_basic_values(data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in the 'status' column of the given DataFrame with 'ok'.
        This function is intended to fill NA values in the 'status' column with the string 'ok',
        typically used when a player has played and their status is not otherwise specified.
        Args:
            data (pd.DataFrame): The input DataFrame containing a 'status' column.
        Returns:
            pd.DataFrame: The DataFrame with missing 'status' values filled with 'ok'.
        """


        logger.info("Filling Status with OK...(only NA when player played)")
        data["status"].fillna("ok", inplace=True)

        return data
    
    @staticmethod    
    def filter_last_matches_inference(group: pd.DataFrame) -> pd.DataFrame:
        """
        Filters a DataFrame group to retain only the latest 4 matches based on the 'date' column.
        The function selects the 4 rows with the most recent dates, then sorts them in ascending order by date.
        Args:
            group (pd.DataFrame): A pandas DataFrame containing match data with a 'date' column.
        Returns:
            pd.DataFrame: A DataFrame containing only the latest 4 matches, sorted by date in ascending order.
        """


        logger.info("Filtering for only the latest matches...")
        group = group.nlargest(4, 'date').sort_values(by='date', ascending=True)
        return group
    
    @staticmethod    
    def fill_market_price(group: pd.DataFrame, training: bool) -> pd.DataFrame:
        """
        Fills missing player price values in a DataFrame using interpolation and fallback strategies.
        Parameters
        ----------
        group : pd.DataFrame
            DataFrame containing player price information, expected to have columns such as
            'fixed_round', 'player_price', 'player_price_for_match', and 'player_price_now'.
        training : bool
            If True, fills missing values in 'player_price' for training data.
            If False, fills missing values in 'player_price_for_match' for inference data.
        Returns
        -------
        pd.DataFrame
            DataFrame with missing price values filled using linear interpolation, forward/backward fill,
            and fallback values depending on the context (training or inference).
        Notes
        -----
        - For training data, remaining NaNs are filled with a minimum value (150,000).
        - For inference data, remaining NaNs are filled with the current player price ('player_price_now').
        - The price values are rounded and cast to integers before returning.
        """


        logger.info("Filling NA marker prices with linear interpolation or repetiton...")

        if training:
            price_field = "player_price"
        elif not training:
            price_field = "player_price_for_match"    

        group = group.sort_values('fixed_round', ascending=True) #The order should already be like this

        if not training:
            # For inference, if we dont have last price it's best to fill it with price_now to avoid false exagerated price changes
            group.loc[(group["fixed_round"] == group["fixed_round"].max()) & (group[price_field].isna()), price_field] = group["player_price_now"]

        # Interpolate linearly for internal missing values
        group[price_field] = group[price_field].interpolate(method='linear')

        # For any remaining NaNs at start or end, fill with nearest known value
        group[price_field] = group[price_field].ffill().bfill()

        if training:
            # Remaining with 150K, minimum value (for players that didn't play one minute all season)
            group[price_field] = group[price_field].fillna(150000)
        elif not training:
            group[price_field] = group[price_field].fillna(group["player_price_now"])
    
        group[price_field] = group[price_field].round().astype(int)

        return group
    
    @staticmethod
    def curate_and_simplify_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters and simplifies player features in the provided DataFrame.
        This function performs the following steps:
        1. Removes rows where 'player_position' is 5 (coaches).
        2. Maps the 'status' column to simplified values using Features.status_map.
        3. Drops the original 'status' column.
        Args:
            data (pd.DataFrame): Input DataFrame containing player features.
        Returns:
            pd.DataFrame: Filtered and simplified DataFrame.
        """

        logger.info(f"Removing position = 5, as they are coaches...")
        data_filtered = data[data["player_position"].isin([1,2,3,4])]

        logger.info(f"Simplifying status...")
        data_filtered["status_mapped"] = data_filtered["status"].map(Features.status_map)

        data_filtered = data_filtered.drop(columns=['status'])

        return data_filtered

    
    @staticmethod    
    def prefilter_features_to_use(data: pd.DataFrame, training: bool) -> pd.DataFrame:
        """
        Filters the input DataFrame to retain only the preselected features for either training or inference.
        Args:
            data (pd.DataFrame): The input DataFrame containing all features.
            training (bool): If True, selects features for training; if False, selects features for inference.
        Returns:
            pd.DataFrame: The filtered DataFrame containing only the selected features.
        Logs:
            Info message indicating the start of feature preselection.
        """

        logger.info(f"Preselecting features...")
        if training:
            data = data[Features.preselected_features_training] 
        elif not training:
            data = data[Features.preselected_features_inference]    
        return data
    

    @staticmethod    
    def create_dummies(data: pd.DataFrame, training: bool) -> pd.DataFrame:
        """
        Creates dummy variables for categorical features specified in Features.dummy_features.
        This function applies one-hot encoding to the specified columns in the input DataFrame.
        When not in training mode, it ensures that the required dummy columns for the 'status' field
        ("status_mapped_ok", "status_mapped_injured", "status_mapped_sanctioned", "status_mapped_doubt")
        are present in the DataFrame, adding them with default value False if missing.
        Args:
            data (pd.DataFrame): Input DataFrame containing the features.
            training (bool): Flag indicating whether the function is being called during training.
        Returns:
            pd.DataFrame: DataFrame with dummy variables added.
        """

        logger.info(f"Creating dummies for status field...")
        data = pd.get_dummies(data, columns=Features.dummy_features)

        if not training:
            required_cols = ["status_mapped_ok", "status_mapped_injured", "status_mapped_sanctioned", "status_mapped_doubt"]
            for col in required_cols:
                if col not in data.columns:
                    data[col] = False

        return data
    
    @staticmethod
    def past_rolling_avg_features(series: pd.DataFrame, training: bool, past_rows_number: int = 3)-> pd.DataFrame:
        """
        Calculates rolling average features for the past matches in a given series.
        For each row in the input DataFrame, computes the mean of the values from the previous `past_rows_number` rows.
        If there are fewer than 3 previous matches, returns NaN for that row.
        Args:
            series (pd.DataFrame): The input DataFrame containing match data.
            training (bool): If True, uses only past data for rolling average (excluding current row).
                             If False, includes the current row in the rolling window.
            past_rows_number (int, optional): Number of past rows to consider for the rolling average. Default is 3.
        Returns:
            list: A list containing the rolling averages or NaN for each row.
        """

        logger.info(f"Calculating rolling features for avg of last {past_rows_number} matches...")
        results = []
        n = len(series)
        for i in range(n):
            if training:
                window = series.iloc[max(0, i-past_rows_number):i]
            elif not training:
                window = series.iloc[max(0, i+1-past_rows_number):i+1]      
            if len(window) >= 3: #if less than 3 previous matches, data won't be used for model
                results.append(window.mean()) 
            else:
                results.append(np.nan)
        return results
    
    @staticmethod
    def recent_price_change_training(series: pd.DataFrame, past_rows_number: int = 1)-> pd.DataFrame:
        """
        Calculates the price change over a specified number of past rows for each entry in the given DataFrame.
        Args:
            series (pd.DataFrame): The input DataFrame containing price data.
            past_rows_number (int, optional): The number of previous rows to consider for calculating the price change. Defaults to 1.
        Returns:
            pd.DataFrame: A list of price changes for each entry in the DataFrame, based on the specified window size.
        """

        logger.info(f"Calculating price change for last {past_rows_number} matches for training..")
        results = []
        n = len(series)
        for i in range(n):
            window = series.iloc[max(0, i-past_rows_number):i+1]
            results.append(window.iloc[-1] - window.iloc[0]) 
        return results
    
    @staticmethod
    def recent_price_change_inference(group: pd.DataFrame,)-> pd.DataFrame:
        """
        Calculates the recent price change of a player for inference by comparing the current price to the price in the most recent or second most recent match.
        Parameters
        ----------
        group : pd.DataFrame
            A DataFrame containing player match data. Must include 'date', 'player_price_for_match', and 'player_price_now' columns.
        Returns
        -------
        pd.DataFrame
            The input DataFrame with an additional column 'recent_price_change_1' representing the difference between the current player price and the relevant match price.
        Notes
        -----
        - If the time since the last match is greater than 3 days (259200 seconds) or there is only one match, the price from the last match is used.
        - If the time since the last match is less than 3 days, the price from the second to last match is used.
        """

        logger.info(f"Calculating price change since now to last match for inference...")
        date_now = round(time.time())
        date_last_match = group["date"].max()
        how_long_match = date_now - date_last_match

        if how_long_match > 259200 or len(group) == 1: #3 days
            price_match = group.loc[group['date'].idxmax(), 'player_price_for_match'] #last_match
        elif how_long_match < 259200:
            price_match = group.sort_values('date').iloc[-2]['player_price_for_match'] #2_to_last_match

        group['recent_price_change_1'] = group['player_price_now'].iloc[0] - price_match

        return group
    
    @staticmethod
    def matches_date_difference_training(series: pd.DataFrame)-> pd.DataFrame:
        """
        Calculates the time difference between consecutive matches in a given DataFrame for training purposes.
        Args:
            series (pd.DataFrame): A pandas DataFrame containing match date information.
        Returns:
            pd.DataFrame: A list of time differences between consecutive matches.
        """

        logger.info(f"Calculating time passed since last match for training...")
        results = []
        n = len(series)
        for i in range(n):
            window = series.iloc[max(0, i-1):i+1]
            results.append(window.iloc[-1] - window.iloc[0])
        return results
    
    @staticmethod
    def matches_date_difference_inference(group: pd.DataFrame)-> pd.DataFrame:
        """
        Calculates the time difference in seconds between the current time and the last match date in a given group of match dates.
        If the time since the last match is less than 3 days (259200 seconds) or if there is only one match in the group,
        the function calculates the time difference using the second-to-last match date instead.
        Args:
            group (pd.DataFrame): A pandas Series or DataFrame containing match dates as Unix timestamps.
        Returns:
            int: The time difference in seconds between now and the relevant match date.
        """

        logger.info(f"Calculating time passed since last match for inference...")

        date_now = round(time.time())
        date_last_match = group.max()
        how_long_match = date_now - date_last_match
        if how_long_match < 259200 or len(group) == 1: #less than 3 days
            date_2_to_last_match = group.nlargest(2).iloc[-1]
            how_long_match = date_now - date_2_to_last_match

        return how_long_match
    
    @staticmethod
    def price_change_time_ratio(data: pd.DataFrame)-> pd.DataFrame:
        """
        Calculates the ratio of recent price change to the time difference since the last match for each row in the DataFrame.
        Adds a new column 'price_change_time_ratio' to the input DataFrame, where the value is computed as:
        - 0 if 'matches_date_difference' is 0 (to avoid division by zero)
        - Otherwise, 'recent_price_change_1' divided by 'matches_date_difference'
        Args:
            data (pd.DataFrame): Input DataFrame containing 'recent_price_change_1' and 'matches_date_difference' columns.
        Returns:
            pd.DataFrame: The DataFrame with the added 'price_change_time_ratio' column.
        """

        logger.info(f"Calculating price change/time ratio since last match...")
        data["price_change_time_ratio"] =np.where(data["matches_date_difference"] == 0, 0, data["recent_price_change_1"] / data["matches_date_difference"])
        return data
    
    
    @staticmethod
    def future_rolling_avg_target_training(series: pd.DataFrame, future_rows_number: int = 1)-> pd.DataFrame:
        """
        Calculates the rolling average of future target values for training purposes.
        For each row in the input DataFrame, computes the mean of the target values in the next `future_rows_number` rows.
        If there are fewer than `number_clipped_rows` future rows available, returns NaN for that position.
        Parameters
        ----------
        series : pd.DataFrame
            The input DataFrame containing the target values.
        future_rows_number : int, optional
            The number of future rows to include in the rolling average calculation (default is 1).
        Returns
        -------
        list
            A list containing the rolling future averages or NaN where insufficient future data is available.
        Notes
        -----
        - The value of `number_clipped_rows` is determined based on `future_rows_number`:
            - If `future_rows_number` == 1: `number_clipped_rows` = 1
            - If `future_rows_number` == 3 or 8: `number_clipped_rows` = 3
        - If the window of future rows is smaller than `number_clipped_rows`, the result for that position is NaN.
        - Intended for use in model training where future target averages are required.
        """

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
        """
        Calculates the severity of injuries for training data by identifying consecutive runs of injury events.
        For each entry in the input DataFrame series, the function assigns a value representing the remaining length
        of the current run of consecutive '1's (injury events). Entries with '0' remain as '0'.
        Args:
            series (pd.DataFrame): A pandas DataFrame column or Series containing binary values (0 for no injury, 1 for injury).
        Returns:
            np.ndarray: An array where each '1' in the input is replaced by the remaining length of its consecutive run,
                        and each '0' remains as '0'.
        """

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
        """
        Estimates the number of weeks left until a player's expected return from injury based on a status string.
        The function parses the injury status information, which should contain phrases like
        "Principios de <Mes>", "Mediados de <Mes>", "Finales de <Mes>", or "Vuelta indefinida".
        It calculates the approximate date of return and computes the number of weeks remaining
        from today until that date.
        Args:
            status_info (str): A string describing the injury status, typically in Spanish.
        Returns:
            float: The estimated number of weeks left until the player's return.
        Raises:
            Exception: If the status string cannot be parsed or contains an unknown month or period.
        """

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
        """
        Adds a team strength feature to the input DataFrame based on team rankings.
        This function maps each player's team to a corresponding strength value (rank)
        by merging the input data with a team ranking dataset. The team names are
        normalized and mapped using a predefined mapping. If a team's rank is not found,
        a default value of 4 is assigned.
        Args:
            data (pd.DataFrame): Input DataFrame containing player and match information.
                Must include 'is_player_home', 'home_team', and 'away_team' columns.
        Returns:
            pd.DataFrame: DataFrame with an additional 'player_team_strength' column
                representing the strength (rank) of each player's team.
        """

        logger.info(f"Mapping players to team strength...")

        logger.info(f"Loading team rank info...")
        points_team = pd.read_csv(f"{RAW_DATA_DIR}/{RAW_DATA_TEAM_RANK}", header = None, names = ["team", "rank"])
        points_team["team"] = points_team["team"].str.lower().str.replace(' ', '-', regex=False)
        points_team["team"] = points_team["team"].replace(Features.teams_map)

        logger.info(f"Adding new team strength feature...")

        data['team'] = np.where(data['is_player_home'], data['home_team'], data['away_team'])

        data_new_feature = data.merge(points_team, on='team', how='left')

        data_new_feature = data_new_feature.rename(columns={"rank": "player_team_strength"})

        data_new_feature = data_new_feature.fillna(4)

        return data_new_feature

    
    @staticmethod
    def remove_nans_for_rolling_avgs(data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows from the input DataFrame where the 'prediction_target_puntuacion_media_roll_avg' column contains NaN values.
        Args:
            data (pd.DataFrame): Input DataFrame containing the 'prediction_target_puntuacion_media_roll_avg' column.
        Returns:
            pd.DataFrame: DataFrame with rows containing NaN values in 'prediction_target_puntuacion_media_roll_avg' removed.
        """

        logger.info(f"Removing NANs for target rolling avg...")

        data = data.dropna(subset=["prediction_target_puntuacion_media_roll_avg"])

        return data
    
    @staticmethod    
    def final_features_select(data: pd.DataFrame, training: bool) -> pd.DataFrame:
        """
        Selects the final set of features from the input DataFrame based on the mode (training or inference).
        Args:
            data (pd.DataFrame): The input DataFrame containing all features.
            training (bool): If True, selects features for training; otherwise, selects features for inference.
        Returns:
            pd.DataFrame: DataFrame containing only the selected features for the specified mode.
        Logs:
            Info message indicating the start of the final feature selection process.
        """

        logger.info(f"Final selection of features...")
        if training:

            data = data[Features.final_selected_features_training]

        elif not training:

            data = data[Features.final_selected_features_inference]

        return data
    
    @staticmethod
    def get_only_last_match_inference(data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the input DataFrame to retain only the last match for each player, 
        then updates specific columns.
        Parameters:
            data (pd.DataFrame): Input DataFrame containing player match data. 
                Must include columns 'player', 'date', 'player_price_now', and 'fixed_round'.
        Returns:
            pd.DataFrame: DataFrame containing only the last match for each player, 
                with 'player_price' set to 'player_price_now' and 'fixed_round' incremented by 1.
        """


        data = data.loc[data.groupby("player")["date"].idxmax()]
        data["player_price"] = data["player_price_now"]
        data["fixed_round"] = data["fixed_round"] + 1

        return data


if __name__ == "__main__":
    app()
