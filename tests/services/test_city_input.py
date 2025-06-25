from app.schemas.schemas import CityInput, Prediction

city_input = CityInput(
    surface_bati=35,
    nombre_pieces=4,
    type_local="house",
    surface_terrain=85,
    nombre_lots=0
)

class TestCityInput:

    def test_get_prediction(self):
        result = city_input.get_prediction("lille")
        assert isinstance(result, Prediction)
        assert isinstance(result.prix_m2_estime, str)
        assert result.ville_modele.lower() == "lille"
        assert isinstance(result.model, str)