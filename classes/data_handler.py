import operator
from abc import ABC
import os
from functools import partial
from time import sleep

import pandas as pd
from geopy import Nominatim


class Filter():
    def __init__(self, field, comparator, value):
        self.field = field
        self.comparator = comparator
        self.value = value

class DataHandler(ABC):

    @classmethod
    def extract_data(cls, filepath: str, filters: list[Filter]) -> pd.DataFrame:
        """
        Extracts data from a file using filtering options.

        :param filepath: file to extract data from.
        :param cities: cities for which to keep data.
        :param mutation_types: mutations for which to keep data.
        :param required_fields: fields that can not be null.

        :return: a dataframe containing data extracted from the file.
        """
        ops = {
            '==': operator.eq,
            '!=': operator.ne,
            '>': operator.gt,
            '>=': operator.ge,
            '<': operator.lt,
            '<=': operator.le,
            'in': lambda series, val: series.isin(val),
            'not in': lambda series, val: ~series.isin(val),
            'contains': lambda series, val: series.str.contains(val, case=False, na=False),
            'startswith': lambda series, val: series.str.startswith(val, na=False),
            'endswith': lambda series, val: series.str.endswith(val, na=False),
            'notnull': lambda series, val: series.notna() & series.astype(str).str.strip().ne('') & series.astype(str).str.lower().ne('none') & series.astype(str).str.lower().ne('nan'),
        }

        df = pd.read_csv(filepath, sep='|', low_memory=False)

        mask = pd.Series([True] * len(df))

        for f in filters:
            if f.comparator not in ops:
                raise ValueError(f"Comparator '{f.comparator}' not supported.")

            # Case insensitive
            col = df[f.field]
            val = f.value

            if pd.api.types.is_string_dtype(col) or col.dtype == object:
                col = col.astype(str).str.strip().str.lower()
                if isinstance(val, str):
                    val = val.strip().lower()

            op = ops[f.comparator]
            mask &= op(col, val)

        df_filtered = df[mask].copy()

        print(f"Data extracted")
        return df_filtered

    @classmethod
    def add_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts values to float and adds a prix_m2 column.

        :param df: dataframe to manipulate.

        :return: a dataframe with converted values.
        """
        valeur_fonciere = df['Valeur fonciere'].astype(str).str.replace(',', '.').str.replace(' ', '').astype(float)
        surface_reelle_bati = df['Surface reelle bati'].astype(str).str.replace(',', '.').str.replace(' ', '').astype(float)
        df['Nombre pieces principales'] = df['Nombre pieces principales'].astype(str).str.replace(',', '.').str.replace(' ', '').astype(float)

        print("Data converted")

        df['prix_m2'] = valeur_fonciere / surface_reelle_bati

        print("Dataset upgraded!")
        return df

    @classmethod
    def add_geolocalization(cls, df: pd.DataFrame) -> pd.DataFrame:
        geolocator = Nominatim(user_agent="geo_immobilier")

        df['adress'] = (
                df['No voie'].fillna(0).astype(int).astype(str).fillna('') + ' ' +
                df['Type de voie'].fillna('RUE') + ' ' +
                df['Voie'] + ', ' +
                df['Code postal'].astype(int).astype(str) + ' ' +
                df['Commune']
        )

        def geocode_address(address):
            try:
                location = geolocator.geocode(address)
                if location:
                    print(location.address)
                    print((location.latitude, location.longitude))
                    return pd.Series([location.latitude, location.longitude])
            except Exception as e:
                print(f"Error for {address}: {e}")
            sleep(1)
            return pd.Series([None, None])

        df[['latitude', 'longitude']] = df['adress'].apply(geocode_address)

        print("Geolocalization added!")
        return df

    @classmethod
    def clean_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans faulty rows from dataframe.

        :param df: dataframe to clean.

        :return: a cleaned dataframe.
        """
        mask = df.notna().all(axis=1) & df.apply(
            lambda col: col.astype(str).str.strip().ne('')).all(axis=1)

        df_cleaned = df[mask].copy()

        outliers = cls.get_outliers(df_cleaned, 'prix_m2')
        df_cleaned = df_cleaned[~outliers].copy()

        print("Data cleaned")
        return df_cleaned

    @classmethod
    def persist_data(cls, df: pd.DataFrame, filepath: str) -> None:
        """
        Persists a dataframe in a csv file.

        :param df: dataframe to persist.
        :param filepath: file to persist to.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

        print(f"Data persisted to {filepath}")

    @classmethod
    def read_data(cls, filepath: str):
        return pd.read_csv(filepath, sep=',', low_memory=False)

    @classmethod
    def get_outliers(cls, df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return (df[column] < lower_bound) | (df[column] > upper_bound)