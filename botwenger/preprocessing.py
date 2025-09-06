from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from botwenger.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import tarfile
import pandas as pd

app = typer.Typer()

class Preprocessing:

    @staticmethod
    def loading_raw_data(path: str) -> pd.DataFrame:
        logger.info("Uncompressing files and loading into DataFrame...")
        dfs = []
        with tarfile.open(path, "r:gz") as tar:
            for member in tqdm(tar.getmembers(), desc="Extracting files"):
                if member.isfile() and member.name.endswith(".csv"):
                    f = tar.extractfile(member)
                    if (f is not None):
                        try:
                            df = pd.read_csv(f)
                            dfs.append(df)
                        except pd.errors.EmptyDataError:
                            #logger.info(f"{member.name} is empty. Passing")
                            pass    
        data = pd.concat(dfs, ignore_index=True)
        logger.success("Uncompressing files completed.")
        return data
    
    @staticmethod
    def basic_parsing(data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Parsing fields...")

        cols_to_int = ["away_team_goals", "home_team_goals", "league_round", "minutes_played", 
                       "player_assists", "player_penalti_goals", "player_non_penalti_goals",
                       "puntuacion_media_sofascore_as"]

        for col in cols_to_int:
            data[col] = data[col].astype(int)

        cols_to_bool = ["player_red_card", "player_second_yellow"]

        for col in cols_to_bool:
            data[col] = data[col].astype(bool)

        logger.info("Parsed fields")

        return data





if __name__ == "__main__":
    app()
