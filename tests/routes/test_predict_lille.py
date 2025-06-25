from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_lille():

    data = {
        "surface_bati": 12,
        "nombre_pieces": 25,
        "type_local": "house",
        "surface_terrain": 20,
        "nombre_lots": 5
    }

    resp = client.post("/predict/lille", json=data)
    assert resp.status_code == 200
    assert isinstance(resp.json()['prix_m2_estime'], str)
    assert resp.json()['ville_modele'] == "Lille"
    assert isinstance(resp.json()['model'], str)

def test_predict_lille_wrong_type():

    data = {
        "surface_bati": 12,
        "nombre_pieces": 25,
        "type_local": "cabinet",
        "surface_terrain": 20,
        "nombre_lots": 5
    }

    resp = client.post("/predict/lille", json=data)
    assert resp.status_code == 422
    assert resp.json() == {
    "erreurs": [
        {
            "champ": "type_local",
            "message": "Input should be 'house' or 'apartment'"
        }
    ]
}