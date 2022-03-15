from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from typing import Optional

from joblib import load
import pandas as pd
import numpy as np

app = FastAPI()

# Chargement des modèles de prédiction
model_RF = load('model_rf_oversampled.joblib')
model_KNN = load('model_knn_oversampled.joblib')

# Liste des catégories dans pnns_groups_2
list_pnns_groups_2 = load('list_pnns_grps_2.joblib')

templates = Jinja2Templates(directory="templates")

### Modèle ###

# Fonction de prédiction avec le modèle Random Forest


def predict_RF(energy_100g, fat_100g, saturated_fat_100g, carbohydrates_100g, sugars_100g, fiber_100g, proteins_100g, salt_100g, fruits_vegetables_nuts_estimate_from_ingredients_100g, pnns_group_2):
    df_v2 = pd.DataFrame(np.array([[energy_100g, fat_100g, saturated_fat_100g, carbohydrates_100g, sugars_100g, fiber_100g, proteins_100g, salt_100g, fruits_vegetables_nuts_estimate_from_ingredients_100g, pnns_group_2]]),
                         columns=['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'fruits-vegetables-nuts-estimate-from-ingredients_100g', 'pnns_groups_2'])
    return model_RF.predict(df_v2)[0]

# Fonction de prédiction avec le modèle KNN


def predict_KNN(energy_100g, fat_100g, saturated_fat_100g, carbohydrates_100g, sugars_100g, fiber_100g, proteins_100g, salt_100g, fruits_vegetables_nuts_estimate_from_ingredients_100g, pnns_group_2):
    df_v2 = pd.DataFrame(np.array([[energy_100g, fat_100g, saturated_fat_100g, carbohydrates_100g, sugars_100g, fiber_100g, proteins_100g, salt_100g, fruits_vegetables_nuts_estimate_from_ingredients_100g, pnns_group_2]]),
                         columns=['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'fruits-vegetables-nuts-estimate-from-ingredients_100g', 'pnns_groups_2'])
    return model_KNN.predict(df_v2)[0]

# Récupère formulaire


@app.get("/")
def form_v2(request: Request):
    list_model = ["Random Forest", "KNN"]
    return templates.TemplateResponse('form.html', context={'request': request, "list_pnns_groups_2": list_pnns_groups_2, "list_model": list_model})

# Effectue la prédiction et affiche résultat sur page grade.html


@app.post("/")
def form_post(request: Request, model: str = Form(...), energy_100g: float = Form(...), fat_100g: float = Form(...), saturated_fat_100g: float = Form(...), carbohydrates_100g: float = Form(...), sugars_100g: float = Form(...), fiber_100g: float = Form(...), proteins_100g: float = Form(...), salt_100g: float = Form(...), fruits_vegetables_nuts_estimate_from_ingredients_100g: float = Form(...), pnns_groups_2: str = Form(...)):
    nutriscore_grade = ""
    if model == "Random Forest":
        nutriscore_grade = predict_RF(energy_100g, fat_100g, saturated_fat_100g, carbohydrates_100g, sugars_100g,
                                      fiber_100g, proteins_100g, salt_100g, fruits_vegetables_nuts_estimate_from_ingredients_100g, pnns_groups_2)
        nutriscore_grade = 'z'
    elif model == "KNN":
        nutriscore_grade = predict_KNN(energy_100g, fat_100g, saturated_fat_100g, carbohydrates_100g, sugars_100g,
                                       fiber_100g, proteins_100g, salt_100g, fruits_vegetables_nuts_estimate_from_ingredients_100g, pnns_groups_2)
    return templates.TemplateResponse('grade.html', context={'request': request, 'nutriscore_grade': nutriscore_grade})


### API REST ###


class Item(BaseModel):
    model: str = "Random Forest"
    energy_100g: float = 0.0
    fat_100g: float = 0.0
    saturated_fat_100g: float = 0.0
    carbohydrates_100g: float = 0.0
    sugars_100g: float = 0.0
    fiber_100g: float = 0.0
    proteins_100g: float = 0.0
    salt_100g: float = 0.0
    fruits_vegetables_nuts_estimate_from_ingredients_100g: float = 0.0
    pnns_groups_2: str = "unknown"


@app.post("/api/predict")
def read_nutriscore(item: Item):
    if item.model == "Random Forest":
        result = predict_RF(item.energy_100g, item.fat_100g, item.saturated_fat_100g, item.carbohydrates_100g, item.sugars_100g, item.fiber_100g,
                            item.proteins_100g, item.salt_100g, item.fruits_vegetables_nuts_estimate_from_ingredients_100g, item.pnns_groups_2)
        result = 'z'
    elif item.model == "KNN":
        result = predict_KNN(item.energy_100g, item.fat_100g, item.saturated_fat_100g, item.carbohydrates_100g, item.sugars_100g, item.fiber_100g,
                             item.proteins_100g, item.salt_100g, item.fruits_vegetables_nuts_estimate_from_ingredients_100g, item.pnns_groups_2)

    predict_json = {
        "model": item.model,
        "nutriscore_grade": result,
    }
    return JSONResponse(content=predict_json)

# http http://127.0.0.1:8000/api/predict energy_100g=1 fat_100g=1 saturated_fat_100g=1 carbohydrates_100g=1 sugars_100g=1 fiber_100g=1 proteins_100g=1 salt_100g=1 fruits_vegetables_nuts_estimate_from_ingredients_100g=1 pnns_groups_2="Meat"

# 812.0    0.0    0.0    47.22    41.67    2.8    0.00    0.9025    0.000000    Dressings and sauces    d
# 1954.0    43.33    16.67    3.33    0.0    0.0    20.00    4.4175    0.0    Processed meat    e
