import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_predict_success():

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

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json=data)
        assert resp.status_code == 200
        assert isinstance(resp.json()['prix_m2_estime'], float)
        assert resp.json()['ville_modele'] == "Lille"
        assert isinstance(resp.json()['model'], str)

@pytest.mark.asyncio
async def test_predict_failure():

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

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json=data)
        assert resp.status_code == 404
        assert resp.json() == {
            "detail": "City not supported"
        }