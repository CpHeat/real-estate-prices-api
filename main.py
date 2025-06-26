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
                # Filter('Nombre pieces principales', '==', 4),
            ]

            df = DataHandler.extract_data("data/ValeursFoncieres-2022.txt", filters)
            if type_local == "maison":
                df = df[["Surface reelle bati", "Nombre pieces principales", "Type local", "Surface terrain", "Nombre de lots", "Valeur fonciere", "No voie", "Type de voie", "Voie", "Code postal", "Commune"]]
            else:
                df = df[["Surface reelle bati", "Nombre pieces principales", "Type local", "Nombre de lots", "Valeur fonciere", "No voie", "Type de voie", "Voie", "Code postal", "Commune"]]
            df = DataHandler.add_data(df)
            df = DataHandler.clean_data(df)
            DataHandler.persist_data(df, f"data/{city}_{type_local}.csv")

def clean_data(filepath: str):
    df = DataHandler.read_data(f"data/{filepath}")
    df = DataHandler.clean_data(df)
    DataHandler.persist_data(df, f"data/cleaned {filepath}")

def train_model(cities: list[str], comparison_city:str, types_local: list[str], model_type, tested_parameters: dict = None):
    evaluation_results = {}
    models = {}

    for city in cities:
        for type_local in types_local:
            df = DataHandler.read_data(f"data/cleaned {city}_{type_local}.csv")
            df_comparison = DataHandler.read_data(f"data/cleaned {comparison_city}_{type_local}.csv")
            model = Model(df, df_comparison)
            model.clean_outliers("prix_m2")
            if type_local == "maison":
                model.set_data(["Surface reelle bati", "Surface terrain", "Nombre pieces principales", "Nombre de lots"])
            else:
                model.set_data(["Surface reelle bati", "Nombre pieces principales", "Nombre de lots"])

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

    extract_data(["lille", "bordeaux"], ["appartement", "maison"])
    clean_data("lille_maison.csv")
    clean_data("lille_appartement.csv")
    clean_data("bordeaux_maison.csv")
    clean_data("bordeaux_appartement.csv")

    evaluation_results = {}
    models = {}

    decision_tree_results = train_model(["lille"],
        "bordeaux",
        ["maison", "appartement"],
        DecisionTreeRegressor)

    evaluation_results['DecisionTree'] = decision_tree_results['results']
    models['DecisionTree'] = decision_tree_results['models']

    random_forest_results = train_model(
        ["lille"],
        "bordeaux",
        ["maison", "appartement"],
        RandomForestRegressor
    )

    evaluation_results['RandomForest'] = random_forest_results['results']
    models['RandomForest'] = random_forest_results['models']

    linear_regression_results = train_model(
        ["lille"],
        "bordeaux",
        ["maison", "appartement"],
        LinearRegression
    )

    evaluation_results['LinearRegression'] = linear_regression_results['results']
    models['LinearRegression'] = linear_regression_results['models']

    xgboost_optimized_results = train_model(
        ["lille"],
        "bordeaux",
        ["maison", "appartement"],
        XGBRegressor,
        {
            'learning_rate': [0.1],
            'gamma': [1],
            'max_depth': [6],
            'min_child_weight': [5],
            'max_delta_step': [0],
            'subsample': [0.5],
            'sampling_method': ['uniform'],
            'colsample_bytree': [1],
            'colsample_bylevel': [1],
            'colsample_bynode': [1],
            'reg_lambda': [1],
            'reg_alpha': [0.5],
            'n_estimators': [10]
        }
    )

    evaluation_results['XgboostOptimized'] = xgboost_optimized_results['results']
    models['XgboostOptimized'] = xgboost_optimized_results['models']


    ResultsHandler.show_metrics_comparison(evaluation_results)
    best_models = ResultsHandler.get_best_model(evaluation_results)

    for best_model in best_models:
        models[best_model['Model']][f"{best_model['City']} {best_model['Type']}"].persist(f"{best_model['Model']} {best_model['City']} {best_model['Type']}")