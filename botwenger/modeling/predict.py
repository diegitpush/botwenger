from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pulp
import pandas as pd

from botwenger.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

class Predict:

    @app.command()
    @staticmethod
    def main(data: pd.DataFrame, total_budget: int):

        position_dummy_columns = ['player_position_1', 'player_position_2', 'player_position_3', "player_position_4"]

        prices_column = "player_price"
        points_column = "prediction_target_puntuacion_media_roll_avg"
        positions_column = "positions"

        data[positions_column] = data[position_dummy_columns].idxmax(axis=1).str.replace('player_position_', '').astype("int")

        chosen, total_points, total_cost = Predict.knapsack_with_cardinality(
            prices = data[prices_column].values, points = data[points_column].values, positions=data[positions_column].values,
            budget = total_budget, k = 11
            )

        data.iloc[chosen]

    @staticmethod
    def knapsack_with_cardinality(prices, points, positions, budget, k=11):

        logger.info("Starting Knapskack with cardinality problem solver...")

        n = len(prices)
        prob = pulp.LpProblem("knapsack_card", pulp.LpMaximize)

        #decision variables: x[i] = 1 if item i is chosen
        x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n)]

        #objective: maximize total points
        prob += pulp.lpSum(points[i] * x[i] for i in range(n))

        #budget constraint
        prob += pulp.lpSum(prices[i] * x[i] for i in range(n)) <= budget

        #exactly k items
        prob += pulp.lpSum(x[i] for i in range(n)) == k

        #Position constraints TODO INTRODUCIR ALINEACIONES DE PAGO?
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
        prob += pulp.lpSum(x[i] for i in range(n) if positions[i]==4) <= 3    

        #solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        status = pulp.LpStatus[prob.status]

        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"Pulp solver failed. Status: {pulp.LpStatus[status]}")

        chosen = [i for i in range(n) if pulp.value(x[i]) > 0.5]

        total_points = pulp.value(prob.objective)
        total_cost = sum(prices[i] for i in chosen)
        logger.info(f"status: {status}. total_points: {total_points}. total_cost: {total_cost}")

        logger.info("Finished Knapskack with cardinality problem solver")

        return chosen, total_points, total_cost    

if __name__ == "__main__":
    app()
