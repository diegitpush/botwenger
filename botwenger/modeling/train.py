from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
import xgboost as xgb
import shap

from botwenger.config import PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME_1, PROCESSED_DATA_FILENAME_8, MODELS_DIR, MODEL_FILENAME, PROCESSED_DATA_FILENAME_3

app = typer.Typer()

class Train:

    @app.command()
    @staticmethod
    def main(number_matches_to_predict: int = 1):
        """
        Trains an XGBoost regression model to predict player scores based on processed feature data.
        Args:
            number_matches_to_predict (int, optional): Number of matches to predict. Determines which processed data file to use.
                - 1: Uses PROCESSED_DATA_FILENAME_1
                - 3: Uses PROCESSED_DATA_FILENAME_3
                - 8: Uses PROCESSED_DATA_FILENAME_8
        Workflow:
            1. Loads processed feature data based on the number of matches to predict.
            2. Splits data into training, validation, and test sets based on the 'season' column.
            3. Initializes and trains an XGBoost regressor with early stopping using the validation set.
            4. Evaluates model performance on validation and test sets (RMSE and R^2).
            5. Plots SHAP feature importance for the validation set.
            6. Saves the trained model to disk.
        Logs:
            - Training progress and evaluation metrics.
            - Model creation, fitting, and saving steps.
        Saves:
            - Trained model file in the specified models directory.
        """


        logger.info(f"Starting model training with number_matches_to_predict: {number_matches_to_predict}")

        if number_matches_to_predict==1: 
            input_file = PROCESSED_DATA_FILENAME_1
        if number_matches_to_predict==3: 
            input_file = PROCESSED_DATA_FILENAME_3
        elif number_matches_to_predict==8:
            input_file = PROCESSED_DATA_FILENAME_8  

        data = Train.loading_features_data(f"{PROCESSED_DATA_DIR}/{input_file}")

        target_column = "prediction_target_puntuacion_media_roll_avg"
        split_column = "season"
        player_column = "player"
        feature_columns = [col for col in data.columns if col not in [target_column, split_column, player_column]]

        X = data[feature_columns]
        y = data[target_column]

        train_mask = data[split_column].isin([2024,2025])
        test_mask = data[split_column]==2025
        val_mask = data[split_column]==2024

        X_train, y_train = X[~train_mask], y[~train_mask]

        X_val, y_val = X[val_mask], y[val_mask] 

        X_test, y_test = X[test_mask], y[test_mask] 

        logger.info(f"Creating model...")

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            learning_rate=0.1,
            eval_metric="rmse",
            early_stopping_rounds=50,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method="gpu_hist"
        )

        logger.info(f"Fitting model...")

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose = True
        )

        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        logger.info(f"RMSE Val: {rmse:.4f}")
        logger.info(f"R^2 Val: {r2:.4f}")

        Train.shap_feature_importance_plot(model, X_val)

        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"RMSE Test: {rmse:.4f}")
        logger.info(f"R^2 Test: {r2:.4f}")

        output_path = MODEL_FILENAME.replace("[number_matches_to_predict]", str(number_matches_to_predict))

        model.save_model(f"{MODELS_DIR}/{output_path}")

        logger.info(f"Training finished and model saved")


    @staticmethod
    def loading_features_data(path: str) -> pd.DataFrame:
        """
        Loads feature data from a CSV file into a pandas DataFrame.
        Args:
            path (str): The file path to the CSV file containing feature data.
        Returns:
            pd.DataFrame: A DataFrame containing the loaded feature data.
        Logs:
            Info messages indicating the start and completion of data loading.
        """

        logger.info("Loading features data...")
        data = pd.read_csv(path)
        logger.info("Loaded features data")
        return data
    
    @staticmethod
    def shap_feature_importance_plot(model, X_val):
        """
        Generates SHAP feature importance plots for a given model and validation dataset.
        This function computes SHAP values using a TreeExplainer for the provided model and validation data,
        then displays both the summary plot and the bar plot of feature importances.
        Args:
            model: A trained tree-based model compatible with SHAP (e.g., XGBoost, LightGBM, CatBoost).
            X_val (pd.DataFrame or np.ndarray): Validation feature data to compute SHAP values.
        Returns:
            None. Displays SHAP summary and bar plots.
        """


        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        shap.summary_plot(shap_values, X_val)
        shap.summary_plot(shap_values, X_val, plot_type="bar")


if __name__ == "__main__":
    app()
