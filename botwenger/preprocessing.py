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
                            pass    
        data = pd.concat(dfs, ignore_index=True)
        logger.success("Uncompressing files completed.")
        return data




if __name__ == "__main__":
    app()
