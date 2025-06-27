from fastapi import APIRouter, HTTPException
from app.schemas.schemas import CityInput, DynamicInput, Prediction

router = APIRouter()

@router.post("/predict/lille",
    response_model=Prediction|dict,
    response_description="A price estimation",
    tags=["predictions"])
async def predict_lille(user_input: CityInput) -> Prediction|dict:
    """
    Returns a price prediction for a real estate asset in Lille.

    Request body:
    - **surface_bati** (int): Built surface area in square meters.
    - **nombre_pieces** (int): Number of rooms.
    - **type_local** ("house" | "apartment"): Type of property.
    - **surface_terrain** (int, optional): Land surface area. Required if type_local is "house".
    - **nombre_lots** (int): Number of lots.

    Response body:
    - **prix_m2_estime** (float): Estimated price per square meter, in euros.
    - **ville_modele** (str): The city used by the prediction model.
    - **model** (str): The name of the model used for prediction.
    """
    if user_input.type_local.lower() not in ["apartment", "house"]:
        return {"error": "wrong local type"}

    return await user_input.get_prediction("lille")

@router.post("/predict/bordeaux",
    response_model=Prediction|dict,
    response_description="A price estimation",
    tags=["predictions"])
async def predict_bordeaux(user_input: CityInput) -> Prediction|dict:
    """
    Returns a price prediction for a real estate asset in Bordeaux.

    Request body:
    - **surface_bati** (int): Built surface area in square meters.
    - **nombre_pieces** (int): Number of rooms.
    - **type_local** ("house" | "apartment"): Type of property.
    - **surface_terrain** (int, optional): Land surface area. Required if type_local is "house".
    - **nombre_lots** (int): Number of lots.

    Response body:
    - **prix_m2_estime** (float): Estimated price per square meter, in euros.
    - **ville_modele** (str): The city used by the prediction model.
    - **model** (str): The name of the model used for prediction.
    """
    if user_input.type_local.lower() not in ["apartment", "house"]:
        return {"error": "wrong local type"}

    return await user_input.get_prediction("bordeaux")

@router.post("/predict",
    response_model=Prediction|dict,
    response_description="A price estimation",
    tags=["predictions"])
async def predict(user_input: DynamicInput) -> Prediction|dict:
    """
    Returns a price prediction for a real estate asset.

    Request body:
    - **ville** (str): Name of the city.
    - **features** (object): Property details including:
        - **surface_bati** (int): Built surface area in square meters.
        - **nombre_pieces** (int): Number of rooms.
        - **type_local** ("house" | "apartment"): Type of property.
        - **surface_terrain** (int, optional): Land surface area. Required if type_local is "house".
        - **nombre_lots** (int): Number of lots.

    Response body:
    - **prix_m2_estime** (float): Estimated price per square meter, in euros.
    - **ville_modele** (str): The city used by the prediction model.
    - **model** (str): The name of the model used for prediction.
    """
    if user_input.ville.lower() not in ["lille", "bordeaux"]:
        raise HTTPException(status_code=404, detail="City not supported")

    return await user_input.features.get_prediction(user_input.ville)