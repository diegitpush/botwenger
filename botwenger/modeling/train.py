from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import xgboost as xgb

from botwenger.config import PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME_ALPHA 

app = typer.Typer()

class Train:

    @app.command()
    @staticmethod
    def model_alpha():

        data = Train.loading_features_data(f"{PROCESSED_DATA_DIR}/{PROCESSED_DATA_FILENAME_ALPHA}")

        target_column = "prediction_target_puntuacion_media_roll_avg_next_8"
        split_column = "season"
        feature_columns = [col for col in data.columns if col not in [target_column, split_column]]

        X = data[feature_columns]
        y = data[target_column]

        train_mask = data[split_column]!=2025
        val_test_mask = data[split_column]==2025

        X_train, y_train = X[train_mask], y[train_mask]

        X_val_test, y_val_test = X[val_test_mask], y[val_test_mask]

        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, test_size=0.5, random_state=42) 
        
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            learning_rate=0.05,
            eval_metric="rmse",
            early_stopping_rounds=50,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method="gpu_hist"
            )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
            )
        
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R^2: {r2:.4f}")

    @staticmethod
    def loading_features_data(path: str) -> pd.DataFrame:
        logger.info("Loading features data...")
        data = pd.read_csv(path)
        logger.info("Loaded features data")
        return data

if __name__ == "__main__":
    app()
