from statistics import linear_regression

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from classes.data_handler import DataHandler, Filter
from classes.model import Model
from classes.results_handler import ResultsHandler

pd.set_option('display.max_columns', None)


def extract_data(cities: list[str], types_local: list[str]) -> None:
    for city in cities:
        for type_local in types_local:
            filters = [
                Filter('Commune', '==', city),
                Filter('Type local', '==', type_local),
                Filter('Nature mutation', '==', 'vente'),
                Filter('Valeur fonciere', 'notnull', None),
                Filter('Surface reelle bati', 'notnull', None),
                Filter('Nombre pieces principales', '==', 4),
            ]

            df = DataHandler.extract_data("data/ValeursFoncieres-2022.txt", filters)
            if type_local == "maison":
                print("MAISON")
                df = df[["Surface reelle bati", "Nombre pieces principales", "Type local", "Surface terrain", "Nombre de lots", "Valeur fonciere"]]
            else:
                print("AUTRE")
                df = df[["Surface reelle bati", "Nombre pieces principales", "Type local", "Nombre de lots", "Valeur fonciere"]]
            df = DataHandler.add_data(df)
            df = DataHandler.clean_data(df)
            DataHandler.persist_data(df, f"data/{city}_{type_local}.csv")

def clean_data(filepath: str):
    df = DataHandler.read_data(f"data/{filepath}")
    df = DataHandler.clean_data(df)
    DataHandler.persist_data(df, f"data/cleaned {filepath}")

def train_model(cities: list[str], types_local: list[str], model_type, tested_parameters: dict = None):
    evaluation_results = {}
    models = {}

    for city in cities:
        for type_local in types_local:
            df = DataHandler.read_data(f"data/cleaned {city}_{type_local}.csv")
            model = Model(df)
            model.clean_outliers("prix_m2")
            if type_local == "maison":
                model.set_data(["Surface reelle bati", "Surface terrain", "Nombre pieces principales"])
            else:
                model.set_data(["Surface reelle bati", "Nombre pieces principales"])

            if tested_parameters:
                model.set_optimal_parameters(tested_parameters, model_type)

            model.train_model(model_type)
            evaluation_results[f"{city} {type_local}"] = model.get_predict_results()
            models[f"{city} {type_local}"] = model

    return {
        "models": models,
        "results": evaluation_results
    }

if __name__ == "__main__":

    # extract_data(["lille"], ["maison", "appartement"])
    # clean_data("lille_maison.csv")
    # clean_data("lille_appartement.csv")

    evaluation_results = {}
    models = {}
    # evaluation_results["linear regression"] = train_model(
    #     ["lille"],
    #     ["appartement", "maison"],
    #     LinearRegression
    # )
    # evaluation_results['decision tree regressor'] = train_model(["lille"],
    #     ["maison", "appartement"],
    #     DecisionTreeRegressor)
    # evaluation_results['random forest regressor'] = train_model(
    #     ["lille"],
    #     ["maison", "appartement"],
    #     RandomForestRegressor
    # )
    #
    # evaluation_results['xgboost'] = train_model(
    #     ["Lille"],
    #     ["maison", "appartement"],
    #     XGBRegressor
    # )
    linear_regression_results = train_model(
        ["Lille"],
        ["maison", "appartement"],
        LinearRegression
    )

    xgboost_optimized_results = train_model(
        ["Lille"],
        ["maison", "appartement"],
        XGBRegressor,
        {
            'n_estimators': [75, 100, 125],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.01, 0.005, 0.02],
            'subsample': [0.75, 0.8, 0.85],
        }
    )

    evaluation_results['xgboost optimized'] = xgboost_optimized_results['results']
    models['xgboost optimized'] = xgboost_optimized_results['models']
    evaluation_results['linear_regression'] = linear_regression_results['results']
    models['linear_regression'] = xgboost_optimized_results['models']

    print(models)

    ResultsHandler.show_metrics_comparison(evaluation_results)