from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
import xgboost as xgb
import shap

from botwenger.config import PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME_1, PROCESSED_DATA_FILENAME_8, PROCESSED_DATA_FILENAME_TEST

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
    def model_predict_points(number_matches_to_predict: int = 1):

        if number_matches_to_predict==1: 
            input_file = PROCESSED_DATA_FILENAME_1
        elif number_matches_to_predict==8:
            input_file = PROCESSED_DATA_FILENAME_8
        else:
            input_file = PROCESSED_DATA_FILENAME_TEST    

        data = Train.loading_features_data(f"{PROCESSED_DATA_DIR}/{input_file}")

        target_column = "prediction_target_puntuacion_media_roll_avg"
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
