from statistics import linear_regression

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from classes.data_handler import DataHandler, Filter
from classes.model import Model
from classes.results_handler import ResultsHandler
from classes.project_settings import ProjectSettings

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
                if project_settings.geolocalize:
                    df = df[["Surface reelle bati", "Nombre pieces principales", "Type local", "Surface terrain", "Nombre de lots", "Valeur fonciere", "No voie", "Type de voie", "Voie", "Code postal", "Commune"]]
                else:
                    df = df[["Surface reelle bati", "Nombre pieces principales", "Type local", "Surface terrain", "Nombre de lots", "Valeur fonciere"]]
            else:
                if project_settings.geolocalize:
                    df = df[["Surface reelle bati", "Nombre pieces principales", "Type local", "Nombre de lots", "Valeur fonciere", "No voie", "Type de voie", "Voie", "Code postal", "Commune"]]
                else:
                    df = df[["Surface reelle bati", "Nombre pieces principales", "Type local", "Nombre de lots", "Valeur fonciere"]]

            df = DataHandler.add_data(df)
            if project_settings.geolocalize:
                df = DataHandler.add_geolocalization(df)
            df = DataHandler.clean_data(df)
            DataHandler.persist_data(df, f"data/{city}_{type_local}.csv")

def clean_data(filepath: str):
    df = DataHandler.read_data(f"data/{filepath}")
    df = DataHandler.clean_data(df)
    DataHandler.persist_data(df, f"data/cleaned {filepath}")

def train_model(cities: list[str], types_local: list[str], model_type, tested_parameters: dict = None, comparison_city:str = None):
    evaluation_results = {}
    models = {}

    for city in cities:
        for type_local in types_local:
            df = DataHandler.read_data(f"data/cleaned {city}_{type_local}.csv")

            if comparison_city:
                df_comparison = DataHandler.read_data(f"data/cleaned {comparison_city}_{type_local}.csv")
                model = Model(df, df_comparison)
            else:
                model = Model(df)

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

    project_settings = ProjectSettings()

    # extract_data(["lille", "bordeaux"], ["appartement", "maison"])
    # clean_data("lille_maison.csv")
    # clean_data("lille_appartement.csv")
    # clean_data("bordeaux_maison.csv")
    # clean_data("bordeaux_appartement.csv")

    evaluation_results = {}
    models = {}

    xgboost_optimized_results = train_model(
        ["lille"],
        ["maison"],
        XGBRegressor,
        {
            'learning_rate': [0.1],
            'gamma': [1],
            'max_depth': [7],
            'min_child_weight': [5],
            'max_delta_step': [1],
            'subsample': [0.5],
            'sampling_method': ['uniform'],
            'colsample_bytree': [1],
            'colsample_bylevel': [1],
            'colsample_bynode': [1],
            'reg_lambda': [1],
            'reg_alpha': [1],
            'n_estimators': [10]
        },
        comparison_city="bordeaux"
    )

    evaluation_results['XGBoostOptimized'] = xgboost_optimized_results['results']
    models['XGBoostOptimized'] = xgboost_optimized_results['models']

    random_forest_optimized_results = train_model(
        ["lille"],
        ["appartement"],
        RandomForestRegressor,
        {
            "n_estimators": [100],
            "max_depth": [1],
            "min_samples_split": [2],
            "min_samples_leaf": [2],
            "min_weight_fraction_leaf": [0],
            "max_features": [1.0]
        },
        comparison_city="bordeaux"
    )

    evaluation_results['RandomForestOptimized'] = random_forest_optimized_results['results']
    models['RandomForestOptimized'] = random_forest_optimized_results['models']

    ResultsHandler.show_metrics_comparison(evaluation_results)
    best_models = ResultsHandler.get_best_model(evaluation_results)

    for best_model in best_models:
        models[best_model['Model']][f"{best_model['City']} {best_model['Type']}"].persist(f"{best_model['Model']} {best_model['City']} {best_model['Type']}")