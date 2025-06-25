import glob
from typing import Literal, Any

import joblib
import numpy as np
from pydantic import BaseModel


matching_files = glob.glob('./app/models/*lille appartement model*.pkl')
apartment_model = joblib.load(matching_files[0])
matching_files = glob.glob('./app/models/*lille appartement scaler_X*.pkl')
apartment_scaler_X = joblib.load(matching_files[0])
matching_files = glob.glob('./app/models/*lille appartement scaler_y*.pkl')
apartment_scaler_y = joblib.load(matching_files[0])
apartment_model_name = matching_files[0].split(' ')[0].split('\\')[-1]

matching_files = glob.glob('./app/models/*lille maison model*.pkl')
house_model = joblib.load(matching_files[0])
matching_files = glob.glob('./app/models/*lille maison scaler_X*.pkl')
house_scaler_X = joblib.load(matching_files[0])
matching_files = glob.glob('./app/models/*lille maison scaler_y*.pkl')
house_scaler_y = joblib.load(matching_files[0])
house_model_name = matching_files[0].split(' ')[0].split('\\')[-1]

class Prediction(BaseModel):
    prix_m2_estime: str
    ville_modele: str
    model: str

class CityInput(BaseModel):
    surface_bati: int
    nombre_pieces: int
    type_local: Literal["house", "apartment"]
    surface_terrain: int
    nombre_lots: int

    def get_prediction(self, city: str) -> Prediction:
        if self.type_local.lower() == "apartment":
            X_input = np.array([[self.surface_bati, self.nombre_pieces, self.nombre_lots]])
            X_scaled = apartment_scaler_X.transform(X_input)
            y_scaled = apartment_model.predict(X_scaled)
            y_pred = apartment_scaler_y.inverse_transform(y_scaled.reshape(-1, 1))
            model_name = apartment_model_name
        elif self.type_local.lower() == "house":
            X_input = np.array([[self.surface_bati, self.surface_terrain, self.nombre_pieces, self.nombre_lots]])
            X_scaled = house_scaler_X.transform(X_input)
            y_scaled = house_model.predict(X_scaled)
            y_pred = house_scaler_y.inverse_transform(y_scaled.reshape(-1, 1))
            model_name = house_model_name

        return Prediction(
            prix_m2_estime=f"{round(float(y_pred[0, 0]), 2)}€/m²",
            ville_modele=city.capitalize(),
            model=model_name)

class DynamicInput(BaseModel):
    ville: str
    features: CityInput