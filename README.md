# 🏡 API de Prédiction du Prix au m² Immobilier (Lille & Bordeaux)

Ce projet expose une **API FastAPI** permettant de prédire le **prix immobilier au m²** à **Lille** et **Bordeaux**, à partir de plusieurs modèles d'apprentissage automatique.

Les modèles sont entraînés à partir de données publiques gouvernementales et peuvent effectuer des estimations basées sur les caractéristiques du bien.
Ces données doivent être téléchargées ici : https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/ et copiées dans le dossier data/

---

## 📂 Structure du projet

```
.
├── app/
│   └── models/                 # Modèles et scalers utilisés pour les prédictions
│       ├── RandomForestOptimized lille appartement model.pkl
│       ├── RandomForestOptimized lille appartement scaler_X.pkl
│       ├── RandomForestOptimized lille appartement scaler_y.pkl
│       ├── XGBoostOptimized lille maison model.pkl
│       ├── XGBoostOptimized lille maison scaler_X.pkl
│       └── XGBoostOptimized lille maison scaler_y.pkl
├── routes/                     # API routes
│   └── routes.py
├── schemas/                    # Schémas attendus/renvoyés par l'API
│   └── schemas.py
├── main.py                     # API core + error handling
├── classes/                    # Classes pour la partie entraînement de modèles
│   ├── data_handler.py         # Gestion des données
│   ├── model.py                # Modèles
│   ├── project_settings.py     # Paramètres
│   └── results_handler.py      # Traitement des résultats/comparaison
├── data/
│   └── geolocalized/           # Données géolocalisées
│       ├── bordeaux_appartement_geolocalized.csv
│       ├── bordeaux_maison_geolocalized.csv
│       ├── lille_appartement_geolocalized.csv
│       ├── lille_maison_geolocalized.csv
├── ValeursFoncieres-2022.txt
├── tests/                      # Tests unitaires de l'API
│   ├── routes/
│   │   ├── test_predict.py
│   │   ├── test_predict_bordeaux.py
│   │   └── test_predict_lille.py
│   └── services/
│       └── test_city_input.py
├── .gitignore
├── LICENSE
├── main.py                     # Lancement de la pipeline d'entraînement
├── model training.ipynb        # Notebook d'entraînement de modèles pas à pas
├── pytest.ini                  # Configuration des tests
├── README.md                   # Readme
├── requirements.txt            # Dépendances
```

---

## 🧠 Modèles de machine learning

Deux types de modèles sont utilisés :

- **XGBoost**
- **Random Forest**

Les modèles sont séparés par **type de bien** (`maison` ou `appartement`) mais sont **communs aux deux villes**.

---

## 🗃 Données d’entraînement

Les données proviennent du portail de l'État français :\
[➜ Indicateurs immobiliers par commune (data.gouv.fr)](https://www.data.gouv.fr/fr/datasets/indicateurs-immobiliers-par-commune-et-par-annee-prix-et-volumes-sur-la-periode-2014-2023/)

- **Année utilisée** : 2022
- **Type de données** : Transactions immobilières (prix, surfaces, localisation, etc.)
- **Accès** : Public

---

## 📓 Notebook de traitement et entraînement

Un **notebook Jupyter** est fourni avec le projet pour suivre l’ensemble du **pipeline de modélisation**, de la donnée brute jusqu'à l’export des modèles finaux utilisés par l’API.

### 🔬 Ce notebook permet de :

* Charger et explorer les données publiques de 2022
* Nettoyer les valeurs manquantes ou aberrantes
* Filtrer les transactions pertinentes (uniquement maisons et appartements sur Lille/Bordeaux)
* Créer des jeux de données par type de bien (`maison`, `appartement`)
* Tester plusieurs algorithmes (**XGBoost**, **DecisionTree**, **RandomForest**) avec validation croisée
* Comparer les performances (**MSE**, **RMSE**, **R²**, etc.)
* Sélectionner le meilleur modèle pour chaque type de bien

### 📁 Fichier :

`notebooks/model_training.ipynb`

> 📌 **Astuce** : ce notebook peut être exécuté dans n’importe quel environnement Jupyter compatible (VS Code, Jupyter Lab, Google Colab...) avec les dépendances installées depuis `requirements.txt`.

---

## 🚀 Lancer l'API en local

### 1. Cloner le dépôt

```bash
git clone <url-du-depot>
cd <nom-du-dossier>
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer le serveur

```bash
uvicorn app.main:app --reload
```

L'API sera disponible sur : [http://127.0.0.1:8000](http://127.0.0.1:8000)

Interface Swagger : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)\
Interface ReDoc : [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🛣 Endpoints disponibles

### 🔎 Estimations par ville

#### `POST /predict/lille`

#### `POST /predict/bordeaux`

**Payload JSON attendu :**

```json
{
  "surface_bati": 85,
  "surface_terrain": 50,
  "nombre_pieces": 4,
  "nombre_lots": 2,
  "type_local": "house"
}
```

**Remarques :**

- `surface_terrain` est obligatoire pour les maisons, optionnel pour les appartements.
- `type_local` doit être `house` ou `apartment`

**Exemple curl :**

```bash
curl -X POST http://127.0.0.1:8000/predict/lille \
  -H "Content-Type: application/json" \
  -d '{
        "surface_bati": 100,
        "surface_terrain": 150,
        "nombre_pieces": 5,
        "nombre_lots": 1,
        "type_local": "house"
      }'
```

### 🌐 Endpoint dynamique (ville paramétrable)

#### `POST /predict`

**Payload JSON attendu :**

```json
{
  "ville": "lille",
  "features": {
    "surface_bati": 100,
    "surface_terrain": 150,
    "nombre_pieces": 5,
    "nombre_lots": 1,
    "type_local": "house"
  }
}
```

**Exemple curl :**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "ville": "bordeaux",
        "features": {
          "surface_bati": 80,
          "nombre_pieces": 3,
          "type_local": "apartment",
          "nombre_lots": 1
        }
      }'
```

---

## 🧪 Tests unitaires

Les tests se trouvent dans le dossier `tests/` et peuvent être exécutés avec :

```bash
pytest
```

Un fichier `pytest.ini` est inclus pour la configuration.

---

## 📘 Documentation interactive

Une documentation Swagger est automatiquement générée et accessible depuis :\
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Elle permet de tester directement les endpoints via l'interface web.

---

## 🛠 Technologies utilisées

- Python 3.10+
- FastAPI
- Scikit-learn / XGBoost
- Pandas / NumPy
- Uvicorn
- Pytest

---

## 📄 Licence

Ce projet est sous licence MIT.

---

## 👤 Auteur

Développé par [@CpHeat](https://github.com/CpHeat)

