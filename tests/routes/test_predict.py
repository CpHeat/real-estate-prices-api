from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_success():

    data = {
        "ville": "Lille",
        "features": {
            "surface_bati": 12,
            "nombre_pieces": 25,
            "type_local": "house",
            "surface_terrain": 20,
            "nombre_lots": 5
        }
    }

    resp = client.post("/predict", json=data)
    assert resp.status_code == 200
    assert isinstance(resp.json()['prix_m2_estime'], str)
    assert resp.json()['ville_modele'] == "Lille"
    assert isinstance(resp.json()['model'], str)

def test_predict_failure():

    data = {
        "ville": "Lyon",
        "features": {
            "surface_bati": 12,
            "nombre_pieces": 25,
            "type_local": "house",
            "surface_terrain": 20,
            "nombre_lots": 5
        }
    }

    resp = client.post("/predict", json=data)
    assert resp.status_code == 404
    assert resp.json() == {
        "detail": "City not supported"
    }