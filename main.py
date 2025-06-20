from classes.data_handler import DataHandler

def extract_data(cities: tuple[str, ...]) -> None:
    for city in cities:
        df = DataHandler.extract_data("data/ValeursFoncieres-2022.txt", [city], ["vente"], ["Valeur fonciere", "Surface reelle bati"])
        df = DataHandler.convert_data(df)
        DataHandler.persist_data(df, f"data/{city}.csv")

if __name__ == "__main__":

    extract_data(("lille", "bordeaux"))