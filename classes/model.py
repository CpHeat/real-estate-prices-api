import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class Model():

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.X = None
        self.y = None
        self.X_train_scaled = None
        self.y_train_scaled = None
        self.X_test_scaled = None
        self.y_test_scaled = None
        self.scaler_X = None
        self.scaler_y = None
        self.model = None
        self.parameters = None

    def clean_outliers(self, column: str):

        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)

        self.df = self.df[~outliers].copy()

    def set_data(self, fields: list[str]):
        self.X = self.df[fields].values
        self.y = self.df[["prix_m2"]].values

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.scaler_X = scaler_X = StandardScaler()
        self.scaler_y = scaler_y = StandardScaler()

        self.X_train_scaled = scaler_X.fit_transform(X_train)
        self.y_train_scaled = scaler_y.fit_transform(y_train)
        self.X_test_scaled = scaler_X.transform(X_test)
        self.y_test_scaled = scaler_y.transform(y_test)

    def train_model(self, model_type):
        if self.parameters:
            self.model = model_type(**self.parameters)
        else:
            self.model = model_type()
        self.model.fit(self.X_train_scaled, self.y_train_scaled.ravel())

    def get_predict_results(self):
        y_train_pred_scaled = self.model.predict(self.X_train_scaled)
        mse_train = mean_squared_error(self.y_train_scaled, y_train_pred_scaled)

        y_test_pred_scaled = self.model.predict(self.X_test_scaled)
        mse_test = mean_squared_error(self.y_test_scaled, y_test_pred_scaled)

        return {
            "train results": {
                "MSE": mse_train,
                "RMSE": np.sqrt(mse_train),
                "MAE": mean_absolute_error(self.y_train_scaled, y_train_pred_scaled),
                "R²": r2_score(self.y_train_scaled, y_train_pred_scaled)
            },
            "test results": {
                "MSE": mse_test,
                "RMSE": np.sqrt(mse_test),
                "MAE": mean_absolute_error(self.y_test_scaled, y_test_pred_scaled),
                "R²": r2_score(self.y_test_scaled, y_test_pred_scaled)
            }
        }

    def set_optimal_parameters(self, param_grid: dict, model_type) -> None:

        grid_search = GridSearchCV(model_type(
            objective="reg:squarederror"),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=1
        )
        grid_search.fit(self.X_train_scaled, self.y_train_scaled.ravel())

        print("Best parameters:", grid_search.best_params_)
        self.parameters = grid_search.best_params_

    def persist(self, filepath):

        joblib.dump(self.model, f"models/{filepath}.pkl")
        joblib.dump(self.scaler_X, f"models/{filepath} scaler_X.pkl")
        joblib.dump(self.scaler_y, f"models/{filepath} scaler_y.pkl")