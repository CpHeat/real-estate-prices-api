import asyncio
import glob
from asyncio import to_thread
from typing import Literal, Any

import joblib
import numpy as np
from pydantic import BaseModel, model_validator

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
    prix_m2_estime: float
    ville_modele: str
    model: str

class CityInput(BaseModel):
    surface_bati: int
    nombre_pieces: int
    type_local: Literal["house", "apartment"]
    surface_terrain: int|None = None
    nombre_lots: int

    @model_validator(mode="after")
    def check_surface_terrain_required_for_house(self) -> "CityInput":
        if self.type_local == "house" and (self.surface_terrain is None):
            raise ValueError("Surface terrain required for house")
        return self

    async def get_prediction(self, city: str) -> Prediction:

        def predict(model, scaler_X, scaler_y, ):
            X_scaled = scaler_X.transform(X_input)
            y_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))

            return y_pred

        if self.type_local.lower() == "apartment":
            X_input = np.array([[self.surface_bati, self.nombre_pieces, self.nombre_lots]])
            model_name = apartment_model_name

            y_pred = await asyncio.to_thread(predict, apartment_model, apartment_scaler_X, apartment_scaler_y)

        elif self.type_local.lower() == "house":
            X_input = np.array([[self.surface_bati, self.surface_terrain, self.nombre_pieces, self.nombre_lots]])
            model_name = house_model_name

            y_pred = await asyncio.to_thread(predict, house_model, house_scaler_X, house_scaler_y)

        return Prediction(
            prix_m2_estime= round(float(y_pred[0, 0]), 2),
            ville_modele=city.capitalize(),
            model=model_name)

class DynamicInput(BaseModel):
    ville: str
    features: CityInput