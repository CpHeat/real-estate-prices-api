import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.main import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_predict_lille_success():

    data = {
        "surface_bati": 12,
        "nombre_pieces": 25,
        "type_local": "house",
        "surface_terrain": 20,
        "nombre_lots": 5
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict/lille", json=data)
        assert resp.status_code == 200
        assert isinstance(resp.json()['prix_m2_estime'], float)
        assert resp.json()['ville_modele'] == "Lille"
        assert isinstance(resp.json()['model'], str)

@pytest.mark.asyncio
async def test_predict_lille_wrong_type():

    data = {
        "surface_bati": 12,
        "nombre_pieces": 25,
        "type_local": "cabinet",
        "surface_terrain": 20,
        "nombre_lots": 5
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict/lille", json=data)
        assert resp.status_code == 422
        assert resp.json() == {
        "erreurs": [
            {
                "champ": "type_local",
                "message": "Input should be 'house' or 'apartment'"
            }
        ]
}