from fastapi import APIRouter, HTTPException
from app.schemas.schemas import CityInput, DynamicInput, Prediction

router = APIRouter()


@router.post("/predict/lille",
    response_model=Prediction|dict,
    response_description="A price estimation",
    tags=["predictions"])
def predict_lille(input: CityInput) -> Prediction|dict:

    if input.type_local.lower() not in ["apartment", "house"]:
        return {"error": "wrong local type"}

    return input.get_prediction("lille")

@router.post("/predict/bordeaux",
    response_model=Prediction|dict,
    response_description="A price estimation",
    tags=["predictions"])
def predict_bordeaux(input: CityInput) -> Prediction|dict:

    if input.type_local.lower() not in ["apartment", "house"]:
        return {"error": "wrong local type"}

    return input.get_prediction("bordeaux")

@router.post("/predict",
    response_model=Prediction|dict,
    response_description="A price estimation",
    tags=["predictions"])
def predict(input: DynamicInput) -> Prediction|dict:

    print(input.ville.lower())
    if input.ville.lower() not in ["lille", "bordeaux"]:
        raise HTTPException(status_code=404, detail="City not supported")

    return input.features.get_prediction(input.ville)