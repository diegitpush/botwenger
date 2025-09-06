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
    def fix_league_rounds(group: pd.DataFrame) -> pd.DataFrame:
        # Fixing league rounds
        rounds = group["league_round"].tolist()
        n = len(rounds)

        # get larger than 38 rounds (player moved teams possibly)
        if n > 38:
            group["fixed_round"] = list(range(1, n+1))
            return group

        # Find first non-null
        first_valid_idx = next((i for i, x in enumerate(rounds) if pd.notna(x)), None)
        if first_valid_idx is None:
            group["fixed_round"] = list(range(1, n+1))
            return group

        start_val = int(rounds[first_valid_idx])

        # Create forward sequence
        seq = list(range(start_val - first_valid_idx, start_val - first_valid_idx + n))

        # Adjust if sequence goes out of [1, 38]
        min_val, max_val = min(seq), max(seq)
        shift = 0
        if max_val > 38:
            shift = 38 - max_val
        elif min_val < 1:
            shift = 1 - min_val

        seq = [x + shift for x in seq]
        group["fixed_round"] = seq
        return group



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
