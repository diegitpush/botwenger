from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from botwenger.config import RAW_DATA_DIR, RAW_DATA_FILENAME
import tarfile
import pandas as pd

app = typer.Typer()

class Preprocessing:
    """
    A class containing static methods for preprocessing player history data for training and inference.
    Includes methods for loading raw data from compressed files, fixing league rounds, and filling missing values.
    """

    @app.command()
    @staticmethod
    def preprocessing_training(output_dir: str = "data/interim", is_test: bool = False):

        """
        Preprocesses player history data for training.
        Loads raw data, fixes league rounds, fills missing values, and saves the preprocessed data to a CSV file.
        If is_test is True, returns intermediate dataframes for testing purposes.

        Args:
            output_dir (str): Directory to save the preprocessed CSV file. Defaults to "data/interim".
            is_test (bool): If True, returns intermediate dataframes instead of saving to file.

        Returns:
            dict (optional): If is_test is True, returns a dictionary with intermediate dataframes.
        """
        logger.info("Starting preprocessing for training...")

        raw_data = Preprocessing.loading_raw_data(f"{RAW_DATA_DIR}/{RAW_DATA_FILENAME}")
        fixed_rounds_data = raw_data.groupby(["player", "season"], group_keys=False).apply(Preprocessing.fix_league_rounds)
        filled_minutes_data = Preprocessing.fill_minutes_played_0(fixed_rounds_data)
        filled_puntuacion_data = Preprocessing.fill_puntuacion_media_0(filled_minutes_data)

        preprocessed_data = filled_puntuacion_data.copy()

        if is_test:
            return {
                "raw_data": raw_data,
                "fixed_rounds_data": fixed_rounds_data,
                "filled_data": filled_puntuacion_data,
            }
        
        else:
            preprocessed_data.to_csv(f"{output_dir}/biwenger_players_history_preprocessed.csv", index=False)
            logger.success(f"Finished preprocessing for training. Saved in {output_dir}")


    @app.command()
    @staticmethod
    def preprocessing_inference(raw_data: pd.DataFrame) -> pd.DataFrame:

        """
        Preprocesses player history data for inference.
        Fixes league rounds and fills missing values in the provided raw data.

        Args:
            raw_data (pd.DataFrame): Raw player history data.

        Returns:
            pd.DataFrame: Preprocessed player history data.
        """
        logger.info("Starting preprocessing for inference...")

        fixed_rounds_data = raw_data.groupby(["player", "season"], group_keys=False).apply(Preprocessing.fix_league_rounds)
        filled_minutes_data = Preprocessing.fill_minutes_played_0(fixed_rounds_data)
        filled_puntuacion_data = Preprocessing.fill_puntuacion_media_0(filled_minutes_data)

        preprocessed_data = filled_puntuacion_data.copy()

        logger.success(f"Finished preprocessing for inference")   

        return preprocessed_data 

    @staticmethod
    def loading_raw_data(path: str) -> pd.DataFrame:
        """
        Loads raw player history data from a compressed tar.gz file containing CSV files.

        Args:
            path (str): Path to the tar.gz file.

        Returns:
            pd.DataFrame: Concatenated DataFrame of all CSV files in the archive.
        """
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
        """
        Fixes league round numbering for each player and season group.
        Ensures rounds are sequential and within the valid range [1, 38]. Handles cases where players have more than 38 rounds.

        Args:
            group (pd.DataFrame): DataFrame group for a single player and season.

        Returns:
            pd.DataFrame: DataFrame with a new 'fixed_round' column.
        """
        logger.info("Fixing league rounds... We assume data is ordereded in ascending date order for rounds for each player/season")
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

        logger.success("Fixed league rounds")

        return group

    @staticmethod
    def fill_minutes_played_0(data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in the 'minutes_played' column with 0.

        Args:
            data (pd.DataFrame): DataFrame containing player history data.

        Returns:
            pd.DataFrame: DataFrame with missing 'minutes_played' values filled.
        """
        logger.info("Filling minutes played...")
        data["minutes_played"].fillna(0, inplace=True)
        return data

    @staticmethod
    def fill_puntuacion_media_0(data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in the 'puntuacion_media_sofascore_as' column with 0.

        Args:
            data (pd.DataFrame): DataFrame containing player history data.

        Returns:
            pd.DataFrame: DataFrame with missing 'puntuacion_media_sofascore_as' values filled.
        """
        logger.info("Filling puntuación media...")
        data["puntuacion_media_sofascore_as"].fillna(0, inplace=True)
        return data


if __name__ == "__main__":
    app()
