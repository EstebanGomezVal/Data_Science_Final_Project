import os
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
# Load Champion Model (Pipeline completo)
# -------------------------

def load_champion_pipeline():
    """
    Descarga el modelo champion desde Databricks.
    El modelo champion YA incluye el preprocessor dentro del Pipeline.
    """
    model_uri = f"models:/{MODEL_NAME_UC}@{ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


model = load_champion_pipeline()

# -------------------------
# FastAPI app
# -------------------------

app = FastAPI(
    title="Income Prediction API",
    version="1.0.0",
    description="Servicio que expone el modelo champion desde Databricks MLflow"
)

# -------------------------
# Input Schema
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

    # 1. Convertimos el JSON de entrada en DataFrame
    df = pd.DataFrame([payload.model_dump()])

    # 2. Renombramos columnas a los nombres del entrenamiento
    df = df.rename(columns={
        "education_num": "education.num",
        "marital_status": "marital.status",
        "capital_gain": "capital.gain",
        "capital_loss": "capital.loss",
        "hours_per_week": "hours.per.week",
        "native_country": "native.country"
    })

    # 3. Pasamos el input directamente al modelo.
    #    El modelo incluye preprocessor + modelo final.
    int_cols = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
    for col in int_cols:
        df[col] = df[col].astype("int64")
        
    pred = model.predict(df)

    result = int(pred[0])

    return {
        "prediction": result,
        "class": ">50K" if result == 1 else "<=50K"
    }
