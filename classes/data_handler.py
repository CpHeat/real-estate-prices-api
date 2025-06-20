from abc import ABC
import os

import pandas as pd


class DataHandler(ABC):

    @classmethod
    def extract_data(cls, filepath: str, cities: list[str], mutation_types: list[str], required_fields: list[str]) -> pd.DataFrame:
        """
        Extract data from a file using filtering options.

        :param filepath: file to extract data from.
        :param cities: cities for which to keep data.
        :param mutation_types: mutations for which to keep data.
        :param required_fields: fields that can not be null.

        :return: a dataframe containing data extracted from the file.
        """
        df = pd.read_csv(filepath, sep='|', low_memory=False)

        df_filtered = df[
            df['Commune'].str.lower().isin([city.lower() for city in cities]) &
            df['Nature mutation'].str.lower().isin([mutation_type.lower() for mutation_type in mutation_types]) &
            df[required_fields].notna().all(axis=1)
        ].copy()

        return df_filtered

    @classmethod
    def convert_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert values to float and adds a prix_m2 column.

        :param df: dataframe to manipulate.

        :return: a dataframe with converted values.
        """
        df['Valeur fonciere'] = df['Valeur fonciere'].astype(str).str.replace(',', '.').str.replace(' ', '').astype(float)
        df['Surface reelle bati'] = df['Surface reelle bati'].astype(str).str.replace(',', '.').str.replace(' ', '').astype(float)

        df['prix_m2'] = df['Valeur fonciere'] / df['Surface reelle bati']

        return df

    @classmethod
    def persist_data(cls, df: pd.DataFrame, filepath: str) -> None:
        """
        Persist a dataframe in a csv file.

        :param df: dataframe to persist.
        :param filepath: file to persist to.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)