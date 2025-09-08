from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from botwenger.config import INTERIM_DATA_DIR, INTERIM_DATA_FILENAME

app = typer.Typer()

class Features: 

    @app.command()
    @staticmethod    
    def main():
        pass

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


if __name__ == "__main__":
    app()
