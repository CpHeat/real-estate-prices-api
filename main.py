import pandas as pd

from classes.data_handler import DataHandler, Filter

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
            df = df[["Surface reelle bati", "Nombre pieces principales", "Type local", "Surface terrain", "Nombre de lots", "Valeur fonciere"]]
            df = DataHandler.add_data(df)
            df = DataHandler.clean_data(df)
            DataHandler.persist_data(df, f"data/{city}_{type_local}.csv")

def clean_data(filepath: str):
    df = DataHandler.read_data(f"data/{filepath}")
    df = DataHandler.clean_data(df)
    DataHandler.persist_data(df, f"data/cleaned {filepath}")

if __name__ == "__main__":

    extract_data(["lille"], ["appartement", "maison"])
    clean_data("lille_appartement.csv")
    clean_data("lille_maison.csv")