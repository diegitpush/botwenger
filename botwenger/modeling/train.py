from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import xgboost as xgb
import shap

from botwenger.config import PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME_ALPHA, PROCESSED_DATA_FILENAME_BETA

app = typer.Typer()

class Train:

    @staticmethod
    def loading_features_data(path: str) -> pd.DataFrame:
        logger.info("Loading features data...")
        data = pd.read_csv(path)
        logger.info("Loaded features data")
        return data
    
    @staticmethod
    def shap_feature_importance_plot(model, X_val):

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        shap.summary_plot(shap_values, X_val)
        shap.summary_plot(shap_values, X_val, plot_type="bar")

    @app.command()
    @staticmethod
    def model_alpha():

        data = Train.loading_features_data(f"{PROCESSED_DATA_DIR}/{PROCESSED_DATA_FILENAME_ALPHA}")

        target_column = "prediction_target_puntuacion_media_roll_avg_next_8"
        split_column = "season"
        feature_columns = [col for col in data.columns if col not in [target_column, split_column]]

        X = data[feature_columns]
        y = data[target_column]

        train_mask = data[split_column].isin([2024,2025])
        test_mask = data[split_column]==2025
        val_mask = data[split_column]==2024

        X_train, y_train = X[~train_mask], y[~train_mask]

        X_val, y_val = X[val_mask], y[val_mask] 

        X_test, y_test = X[test_mask], y[test_mask] 
        
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
            verbose=False
            )
        
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        print(f"Alpha RMSE Val: {rmse:.4f}")
        print(f"Alpha R^2 Val: {r2:.4f}")

        Train.shap_feature_importance_plot(model, X_val)

        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Alpha RMSE Test: {rmse:.4f}")
        print(f"Alpha R^2 Test: {r2:.4f}")


    @app.command()
    @staticmethod
    def model_beta():

        data = Train.loading_features_data(f"{PROCESSED_DATA_DIR}/{PROCESSED_DATA_FILENAME_BETA}")

        target_column = "prediction_target_puntuacion_media_roll_avg_next_8"
        split_column = "season"
        feature_columns = [col for col in data.columns if col not in [target_column, split_column]]

        X = data[feature_columns]
        y = data[target_column]

        train_mask = data[split_column].isin([2024,2025])
        test_mask = data[split_column]==2025
        val_mask = data[split_column]==2024

        X_train, y_train = X[~train_mask], y[~train_mask]

        X_val, y_val = X[val_mask], y[val_mask] 

        X_test, y_test = X[test_mask], y[test_mask] 

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            learning_rate=0.05,
            eval_metric="rmse",
            early_stopping_rounds=50,
            max_depth=5,
            min_child_weight=5,
            subsample=0.5,
            colsample_bytree=1,
            random_state=42,
            n_jobs=-1,
            tree_method="gpu_hist"
        )
        
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose = False
        )

    
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        print(f"Beta RMSE Val: {rmse:.4f}")
        print(f"Beta R^2 Val: {r2:.4f}")

        Train.shap_feature_importance_plot(model, X_val)

        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Beta RMSE Test: {rmse:.4f}")
        print(f"Beta R^2 Test: {r2:.4f}")



if __name__ == "__main__":
    app()
