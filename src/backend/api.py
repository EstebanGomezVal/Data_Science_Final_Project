# src/backend/api.py

import os
import pickle
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# -------------------------
# Load env variables
# -------------------------
load_dotenv(override=True)

# -------------------------
# MLflow / Databricks setup
# -------------------------
mlflow.set_tracking_uri("databricks")
client = MlflowClient()

MODEL_NAME_UC = "workspace.default.income-prediction-classifier-prefect"
ALIAS = "champion"

# -------------------------
# Load Champion Model + Preprocessor
# -------------------------

def load_champion_and_preprocessor():
    """
    Descarga desde Databricks:
    - Modelo champion del Model Registry
    - Preprocessor guardado en los artifacts del run
    """

    # 1. Cargar modelo champion desde el registry
    model_uri = f"models:/{MODEL_NAME_UC}@{ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)

    # 2. Obtener run asociado al alias champion
    mv = client.get_model_version_by_alias(MODEL_NAME_UC, ALIAS)
    run_id = mv.run_id

    # 3. Descargar preprocessor desde artifacts
    client.download_artifacts(
        run_id=run_id,
        path="preprocessor",
        dst_path="."      # descarga localmente ./preprocessor/preprocessor.b
    )

    # 4. Cargar el preprocessor
    with open("preprocessor/preprocessor.b", "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


model, preprocessor = load_champion_and_preprocessor()

# -------------------------
# FastAPI app
# -------------------------

app = FastAPI(
    title="Income Prediction API",
    version="1.0.0",
    description="Servicio que expone el modelo champion desde Databricks MLflow"
)

# -------------------------
# Marshmallow / Input Schema
# -------------------------

class IncomeRequest(BaseModel):
    age: float
    workclass: str
    fnlwgt: float
    education: str
    education_num: float
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str


# -------------------------
# Health Check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# Prediction endpoint
# -------------------------

@app.post("/predict")
def predict_endpoint(payload: IncomeRequest):

    # 1. Convert incoming JSON to DataFrame
    df = pd.DataFrame([payload.model_dump()])

    df = df.rename(columns={
        "education_num": "education.num",
        "marital_status": "marital.status",
        "capital_gain": "capital.gain",
        "capital_loss": "capital.loss",
        "hours_per_week": "hours.per.week",
        "native_country": "native.country"
    })

    # 3. Compute prediction (sklearn model)
    pred = model.predict(df)

    result = int(pred[0])

    response = {
        "prediction": result,
        "class": ">50K" if result == 1 else "<=50K"
    }

    return response
