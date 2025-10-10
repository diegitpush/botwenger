from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pulp
import pandas as pd
from datetime import date
import textwrap
import requests
import os
from retry import retry


from botwenger.config import MODELS_DIR, PROCESSED_DATA_DIR, MODEL_FILENAME
from pybiwenger.endpoints import Endpoints
from botwenger.preprocessing import Preprocessing
from botwenger.features import Features

import xgboost as xgb

app = typer.Typer()

class Predict:


    @app.command()
    @staticmethod
    def main():
        Predict.daily_recommended_changes()

    @staticmethod
    def daily_recommended_changes():

        """
        Daily endpoint to:
        - Get/scrape data from our current Biwenger roster + current market
        - Predict future expected points for all these players with a trained XGBoost model
        - Recommend player buys and sells, enforcing budget and roster constraints using a PuLP algorithm
        - Send this information as a beatufied text from a Telegram bot
        """

        preprocessed, balance = Predict.get_and_preprocess_daily_data_api()
        processed = Predict.process_inference_features_data(preprocessed)

        players_value = processed.loc[processed['roster'] == 1, 'player_price'].sum()
        total_budget = players_value + balance

        predictions = Predict.xgboost_model_expected_points_3(processed)

        actual_points = predictions.loc[predictions['roster'] == 1, 'prediction_target_puntuacion_media_roll_avg'].sum()

        if predictions[predictions["roster"] == 1]["roster"].size > 11:
            Predict.choose_starting_11_no_market_and_send(predictions.copy(), players_value, total_budget, "points_only_roster")
        elif predictions[predictions["roster"] == 1]["roster"].size == 11:    
            Predict.choose_all_starting_11s_and_send(predictions.copy(), players_value, total_budget, actual_points, "points") #Optimize for points
            Predict.choose_all_starting_11s_and_send(predictions.copy(), players_value, total_budget, actual_points, "money") #Optimize for money
        elif predictions[predictions["roster"] == 1]["roster"].size < 11:    
            Predict.choose_all_starting_11s_and_send(predictions.copy(), players_value, total_budget, actual_points, "points_incomplete_roster", predictions[predictions["roster"] == 1]["roster"].size) #Optimize for points

    @staticmethod
    def choose_starting_11_no_market_and_send(predictions, players_value, total_budget, optimize_for):

        """
        Chooses the starting 11 sell recommendations for a larger than 11 roster and sends them as a Telegram bot message
        Args:
            predictions (pd.DataFrame): DataFrame containing player information, including price, predicted points, positions, and roster status.
            players_value (int): Total sum of the players value
            total_budget (float): The total budget available for selecting players.
            optimize_for (str): Criterion to optimize for (e.g., 'points', 'value').
        """

        predictions_roster = predictions[predictions["roster"] == 1]

        optimized_only_roster, optimized_only_roster_total_points, optimized_only_roster_total_cost = Predict.choose_starting_11(predictions_roster, total_budget, None, optimize_for)

        players_to_sell = predictions_roster.loc[~predictions_roster['player'].isin(optimized_only_roster["player"])]

        efficiency_team = (optimized_only_roster_total_points/optimized_only_roster_total_cost)*1000000

        money_in_sales = players_value - optimized_only_roster_total_cost

        text = Predict.parse_and_beautify_daily_info_only_roster(players_to_sell, optimized_only_roster_total_cost,
                                                                 optimized_only_roster_total_points, efficiency_team,
                                                                 money_in_sales)
        
        Predict.send_telegram_bot(text)


    @staticmethod
    def choose_all_starting_11s_and_send(predictions, players_value, total_budget, actual_points, optimize_for, roster_size = -1):

        """
        Chooses the starting 11/buy/sell recommendations and sends them as a Telegram bot message
        Args:
            predictions (pd.DataFrame): DataFrame containing player information, including price, predicted points, positions, and roster status.
            players_value (int): Total sum of the players value
            total_budget (float): The total budget available for selecting players.
            actual_points (array-like): Actual points scored by players, used for optimization.
            optimize_for (str): Criterion to optimize for (e.g., 'points', 'value').
            roster_size (int, optional): Size of the roster. Defaults to -1 (no restriction).
        """

        optimized_roster, optimized_total_points, optimized_total_cost = Predict.choose_starting_11(predictions, total_budget, actual_points, optimize_for, roster_size)

        players_to_buy = optimized_roster[~optimized_roster["player"].isin(predictions.loc[predictions['roster'] == 1, 'player'])]

        players_to_sell = predictions.loc[(predictions['roster'] == 1) & (~predictions['player'].isin(optimized_roster["player"]))]
        
        points_gain = optimized_total_points - actual_points
        money_to_spend = optimized_total_cost - players_value

        efficiency_team = (actual_points/players_value)*1000000

        efficiency_recommended_team = (optimized_total_points/optimized_total_cost)*1000000

        if money_to_spend != 0:
            efficiency_recommended_moves = (points_gain/money_to_spend)*1000000
        else:
            efficiency_recommended_moves = "Undefined"

        text = Predict.parse_and_beautify_daily_info(players_to_buy, players_to_sell, optimized_total_cost,
                                      players_value, actual_points,
                                      optimized_total_points, points_gain, money_to_spend,
                                      efficiency_team, efficiency_recommended_moves, 
                                      efficiency_recommended_team, optimize_for)

        Predict.send_telegram_bot(text)  


    @retry(tries=5, delay=10)
    @staticmethod
    def send_telegram_bot(text):

        """
        Sends a Telegram message to a preconfigured bot with the daily recommendations.
        Args:
            text (str): Text to send
        """

        logger.info("Sending Telegram bot message...")

        if (("TELEGRAM_BOT_TOKEN" not in os.environ) or ("TELEGRAM_CHAT_ID" not in os.environ)):
            raise Exception("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env variables")
       
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
       
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "parse_mode": "MarkdownV2", "text": {text}}
       
        response = requests.post(url, data = payload)

        if response.status_code != 200:
            raise Exception(f"Message to Telegram Bot didn't work. Status code: {response.status_code}")

        logger.info("Sent Telegram message!")

    @staticmethod
    def parse_and_beautify_daily_info_only_roster(players_to_sell, optimized_total_cost,
                                      optimized_total_points, efficiency_team, money_in_sales):
        """
        Parses player sell recommendations and optimization metrics for when the roster has more than 11 players, then formats them into a beautified text message.
        Args:
            players_to_sell (pd.DataFrame): DataFrame containing players to sell with columns 'player', 'player_price', and 'prediction_target_puntuacion_media_roll_avg'.
            optimized_total_cost (float): Total cost of the recommended team.
            optimized_total_points (float): Total points of the recommended team.
            money_in_sales (float): Money made in sales.
            efficiency_team (float): Efficiency (points per euro) of the current team.
        Returns:
            str: Beautified and formatted text message summarizing sell recommendations and optimization metrics.
        """
                
        logger.info("Parsing and beautifying text to send for only roster...")

        players_to_sell_tuple = list(players_to_sell[["player","player_price","prediction_target_puntuacion_media_roll_avg"]].itertuples(index=False, name=None)) 

        formatted_sell = "\n        ".join(f"{x} ({round(y):,}€, {round(float(z), 1)} xP3)" for x, y, z in players_to_sell_tuple)

        header = f"*OPTIMIZACIÓN PUNTOS PARA PLANTILLA ({date.today()})*"

        beautiful_text = f"""
        {header}

        *A vender:*
        {formatted_sell}

        *xP3 plantilla recomendada:* {round(float(optimized_total_points), 1)}

        *Valor plantilla recomendada:* {round(optimized_total_cost):,}€
        *Valor delta:* {round(money_in_sales):,}€

        *xP3/€ plantilla recomendada:* {round(float(efficiency_team), 2)}"""

        reformatted_text = textwrap.dedent(beautiful_text).strip().replace("(", "\(").replace(")", "\)").replace("-", "\-").replace(".", "\.")

        return reformatted_text    


    @staticmethod
    def parse_and_beautify_daily_info(players_to_buy, players_to_sell, optimized_total_cost,
                                      players_value, actual_points,
                                      optimized_total_points, points_gain, money_to_spend,
                                      efficiency_team, efficiency_recommended_moves, efficiency_recommended_team,
                                      optimize_for):
        """
        Parses player buy/sell recommendations and optimization metrics, then formats them into a beautified text message.
        Args:
            players_to_buy (pd.DataFrame): DataFrame containing players to buy with columns 'player', 'player_price', and 'prediction_target_puntuacion_media_roll_avg'.
            players_to_sell (pd.DataFrame): DataFrame containing players to sell with columns 'player', 'player_price', and 'prediction_target_puntuacion_media_roll_avg'.
            optimized_total_cost (float): Total cost of the recommended team.
            players_value (float): Total value of the current team.
            actual_points (float): Total points of the current team.
            optimized_total_points (float): Total points of the recommended team.
            points_gain (float): Expected points gain from the recommended moves.
            money_to_spend (float): Expected money difference after recommended moves.
            efficiency_team (float): Efficiency (points per euro) of the current team.
            efficiency_recommended_moves (Union[float, str]): Efficiency (points per euro) of the recommended moves, or "Undefined".
            efficiency_recommended_team (float): Efficiency (points per euro) of the recommended team.
            optimize_for (str): Optimization target, one of "points", "money", or "points_incomplete_roster".
        Returns:
            str: Beautified and formatted text message summarizing buy/sell recommendations and optimization metrics.
        """
                
        logger.info("Parsing and beautifying text to send...")

        players_to_buy_tuple = list(players_to_buy[["player","player_price","prediction_target_puntuacion_media_roll_avg"]].itertuples(index=False, name=None)) 
        players_to_sell_tuple = list(players_to_sell[["player","player_price","prediction_target_puntuacion_media_roll_avg"]].itertuples(index=False, name=None)) 

        formatted_buy = "\n        ".join(f"{x} ({round(y):,}€, {round(float(z), 1)} xP3)" for x, y, z in players_to_buy_tuple)
        formatted_sell = "\n        ".join(f"{x} ({round(y):,}€, {round(float(z), 1)} xP3)" for x, y, z in players_to_sell_tuple)

        if optimize_for == "points" and efficiency_recommended_moves != "Undefined":
            header = f"*OPTIMIZACIÓN PUNTOS ({date.today()})*"
            xp3_move = f"*xP3/€ movimiento:* {round(float(efficiency_recommended_moves), 2)}"

        elif optimize_for == "points" and efficiency_recommended_moves == "Undefined":
            header = f"*OPTIMIZACIÓN PUNTOS ({date.today()})*"
            xp3_move = "*xP3/€ movimiento:* Undefined"

        elif optimize_for == "money":
            header = f"*OPTIMIZACIÓN PRESUPUESTO ({date.today()})*"
            xp3_move = ""

        elif optimize_for == "points_incomplete_roster":
            header = f"*OPTIMIZACIÓN PUNTOS PLANTILLA INCOMPLETA ({date.today()})*"
            xp3_move = ""


        beautiful_text = f"""
        {header}

        *A comprar:*
        {formatted_buy}

        *A vender:*
        {formatted_sell}

        *xP3 plantilla recomendada:* {round(float(optimized_total_points), 1)}
        *xP3 plantilla actual:* {round(float(actual_points), 1)}
        *xP3 delta:* {round(float(points_gain), 1)}

        *Valor plantilla recomendada:* {round(optimized_total_cost):,}€
        *Valor plantilla actual:* {round(players_value):,}€
        *Valor delta:* {round(money_to_spend):,}€

        *xP3/€ plantilla actual:* {round(float(efficiency_team), 2)}
        *xP3/€ plantilla recomendada:* {round(float(efficiency_recommended_team), 2)}
        {xp3_move}"""

        reformatted_text = textwrap.dedent(beautiful_text).strip().replace("(", "\(").replace(")", "\)").replace("-", "\-").replace(".", "\.")

        return reformatted_text
        

    @staticmethod
    def xgboost_model_expected_points_3(data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts expected points for each row in the input DataFrame using a pre-trained XGBoost regression model.
        The function loads a specific XGBoost model (for predicting with 3 matches) and applies it to the input features,
        excluding the columns "player", "season", and "roster". The predicted expected points are added to the DataFrame
        as a new column "prediction_target_puntuacion_media_roll_avg".
        Args:
            data (pd.DataFrame): Input DataFrame containing player data and features for prediction.
        Returns:
            pd.DataFrame: The input DataFrame with an additional column containing the predicted expected points.
        """


        logger.info("Predicting expected points...")

        model_path = MODEL_FILENAME.replace("[number_matches_to_predict]", "3")

        model_3 = xgb.XGBRegressor()
        model_3.load_model(f"{MODELS_DIR}/{model_path}")

        feature_columns = [col for col in data.columns if col not in ["player", "season", "roster"]]

        X = data[feature_columns]

        y_pred = model_3.predict(X)

        data["prediction_target_puntuacion_media_roll_avg"] = y_pred

        logger.info("Predicted expected points")

        return data

    @staticmethod
    def process_inference_features_data(preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Processes preprocessed data to extract inference features.
        Args:
            preprocessed_data (pd.DataFrame): The input DataFrame containing preprocessed data.
        Returns:
            pd.DataFrame: A DataFrame containing the extracted inference features.
        """


        logger.info("Processing inference features...")

        features_data = Features.features_inference(preprocessed_data)

        return features_data


    @staticmethod
    def get_and_preprocess_daily_data_api():
        """
        Retrieves daily data from an API, preprocesses it, and returns the processed data along with the balance.
        The function performs the following steps:
        1. Logs the start of the data retrieval process.
        2. Calls the daily squad market API endpoint to fetch data and balance.
        3. Flattens the nested list structure of the API data.
        4. Converts the flattened data into a pandas DataFrame.
        5. Logs the start of the preprocessing step.
        6. Applies preprocessing suitable for inference on the DataFrame.
        7. Returns the preprocessed data and the balance.
        Returns:
            Tuple[pd.DataFrame, Any]: A tuple containing the preprocessed daily data as a pandas DataFrame and the balance.
        """


        logger.info("Getting daily data from API...")

        api_data, balance = Endpoints.daily_squad_market_endpoint()

        flat_list = [d for inner_list in api_data for d in inner_list]

        flat_data = pd.DataFrame(flat_list)

        logger.info("Preprocessing daily data...")

        preprocessed_data = Preprocessing.preprocessing_inference(flat_data)

        return preprocessed_data, balance


    @staticmethod
    def choose_starting_11(data: pd.DataFrame, total_budget, actual_points, optimize_for, roster_size = -1):
        """
        Selects the optimal starting 11 players from a given dataset based on budget, predicted points, and optimization criteria.
        Args:
            data (pd.DataFrame): DataFrame containing player information, including price, predicted points, positions, and roster status.
            total_budget (float): The total budget available for selecting players.
            actual_points (array-like): Actual points scored by players, used for optimization.
            optimize_for (str): Criterion to optimize for (e.g., 'points', 'value').
            roster_size (int, optional): Size of the roster. Defaults to -1 (no restriction).
        Returns:
            Tuple[pd.DataFrame, float, float]:
                - DataFrame of the selected starting 11 players.
                - Total predicted points of the selected players.
                - Total cost of the selected players.
        """


        position_dummy_columns = ['player_position_1', 'player_position_2', 'player_position_3', "player_position_4"]

        prices_column = "player_price"
        points_column = "prediction_target_puntuacion_media_roll_avg"
        positions_column = "positions"
        roster_column = "roster"

        data[positions_column] = data[position_dummy_columns].idxmax(axis=1).str.replace('player_position_', '').astype("int")

        #We add 10% price to market players for tolerance
        data.loc[data[roster_column] == 0, prices_column] = data.loc[data[roster_column] == 0, prices_column] * 1.1

        chosen, total_points, total_cost = Predict.knapsack_with_cardinality(
            prices = data[prices_column].values, points = data[points_column].values, positions=data[positions_column].values,
            roster = data[roster_column].values, budget = total_budget, actual_points = actual_points, optimize_for = optimize_for, k = 11, roster_size = roster_size)

        data_chosen_11 = data.iloc[chosen]

        return data_chosen_11, total_points, total_cost

    @staticmethod
    def knapsack_with_cardinality(prices, points, positions, roster, budget, actual_points, optimize_for, k=11, roster_size = -1):
        """
        Solves a knapsack problem with cardinality and roster constraints for fantasy football team selection.
        Args:
            prices (list of float): List of player prices.
            points (list of float): List of expected points for each player.
            positions (list of int): List of player positions (1=GK, 2=DF, 3=MC, 4=ST).
            roster (list of bool): List indicating if a player is currently in the roster.
            budget (float): Maximum budget allowed for team selection.
            actual_points (float): Current expected points of the roster (used for 'money' optimization).
            optimize_for (str): Optimization objective. One of "points", "points_only_roster", "points_incomplete_roster", or "money".
            k (int, optional): Number of players to select. Default is 11.
            roster_size (int, optional): Size of the current roster (used for 'points_incomplete_roster'). Default is -1.
        Returns:
            tuple:
                chosen (list of int): Indices of selected players.
                total_points (float): Total expected points of the selected team.
                total_cost (float): Total cost of the selected team.
        Raises:
            RuntimeError: If the solver fails to find an optimal solution.
        Notes:
            - Enforces position constraints: 1 GK, 3-5 DF, 3-5 MC, 1-4 ST.
            - Enforces budget and roster constraints depending on the optimization objective.
            - Uses the PuLP library for solving the integer programming problem.
        """

        logger.info("Starting Knapskack with cardinality problem solver...")

        n = len(prices)
        prob = pulp.LpProblem("knapsack_card", pulp.LpMaximize)

        #decision variables: x[i] = 1 if item i is chosen
        x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n)]


        if optimize_for == "points" or optimize_for == "points_only_roster" or optimize_for == "points_incomplete_roster":
            #objective: maximize total points
            prob += pulp.lpSum(points[i] * x[i] for i in range(n))

            #budget constraint
            prob += pulp.lpSum(prices[i] * x[i] for i in range(n)) <= budget

        elif optimize_for == "money":
            #budget constraint
            prob += pulp.lpSum(prices[i] * x[i] for i in range(n)) <= (budget + 10000) #Tolerance for float precission innacuracies

            #objective: maximize money saving
            prob += pulp.lpSum(-prices[i] * x[i] for i in range(n))

            #maintaining expected points
            prob += pulp.lpSum(points[i] * x[i] for i in range(n)) >= (actual_points - 0.1) #Tolerance for float precission innacuracies


        #exactly k items
        prob += pulp.lpSum(x[i] for i in range(n)) == k

        if optimize_for == "points" or optimize_for == "money":
            #One roster changes max
            prob += pulp.lpSum(x[i] for i in range(n) if roster[i]) >= 10
        elif optimize_for == "points_incomplete_roster":
            #Maintain all current players
            prob += pulp.lpSum(x[i] for i in range(n) if roster[i]) == roster_size


        #Position constraints
        #position GK: 1
        prob += pulp.lpSum(x[i] for i in range(n) if positions[i]==1) == 1

        #position DF: min 3, max 5
        prob += pulp.lpSum(x[i] for i in range(n) if positions[i]==2) >= 3
        prob += pulp.lpSum(x[i] for i in range(n) if positions[i]==2) <= 5

        #position MC: min 3, max 5
        prob += pulp.lpSum(x[i] for i in range(n) if positions[i]==3) >= 3
        prob += pulp.lpSum(x[i] for i in range(n) if positions[i]==3) <= 5

        #position ST: min 1, max 3
        prob += pulp.lpSum(x[i] for i in range(n) if positions[i]==4) >= 1
        prob += pulp.lpSum(x[i] for i in range(n) if positions[i]==4) <= 4

        #solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        status = pulp.LpStatus[prob.status]

        if status != "Optimal":
            raise RuntimeError(f"Pulp solver failed. Status: {status}")

        chosen = [i for i in range(n) if pulp.value(x[i]) > 0.5]

        if optimize_for == "points" or optimize_for == "points_only_roster" or optimize_for == "points_incomplete_roster":
            total_points = pulp.value(prob.objective)
            total_cost = sum(prices[i] for i in chosen)
            logger.info(f"status: {status}. total_points: {total_points}. total_cost: {total_cost}")

        if optimize_for == "money":
            total_cost = -pulp.value(prob.objective)
            total_points = sum(points[i] for i in chosen)
            logger.info(f"status: {status}. total_points: {total_points}. total_cost: {total_cost}")   

        logger.info("Finished Knapskack with cardinality problem solver")

        return chosen, total_points, total_cost    

if __name__ == "__main__":
    app()
