# income_pipeline_prefect.py
import os
import pathlib
import pickle
import mlflow
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from optuna.samplers import TPESampler
import optuna
from mlflow.models.signature import infer_signature
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from prefect import flow, task

# load env
load_dotenv(override=True)

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "/Users/estebangmzv@gmail.com/income-prediction-prefect")
MODEL_NAME_UC = os.getenv("MODEL_NAME_UC", "workspace.default.income-prediction-classifier-prefect")

# set tracking (Databricks expected)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
mlflow.set_experiment(EXPERIMENT_NAME)


# Cargar y preprocesar datos
@task(name="Load and Preprocess Data")
def load_and_preprocess(file_path: str, random_state: int = 42):
    """Return both raw splits (X_train_raw, ...) and processed splits (X_train, ...) plus y and preprocessor."""
    df = pd.read_csv(file_path)

    # target
    y = df["income"].apply(lambda x: 1 if ">50K" in x else 0)
    X = df.drop(["income", "education"], axis=1)

    # drop indice column si existe
    X = X.drop(columns=["index"], errors="ignore")

    categorical_cols = [
        'workclass', 'marital.status', 'occupation', 'race',
        'relationship', 'sex', 'native.country'
    ]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        [
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    # splits (estratificados)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # fit/transform using only training raw
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_val_proc = preprocessor.transform(X_val_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    # feature names 
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_train_proc.shape[1])]

    X_train = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train_raw.index)
    X_val = pd.DataFrame(X_val_proc, columns=feature_names, index=X_val_raw.index)
    X_test = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test_raw.index)

    # preprocessor local
    pathlib.Path("preprocessor").mkdir(exist_ok=True)
    with open("preprocessor/preprocessor.b", "wb") as f_out:
        pickle.dump(preprocessor, f_out)

    # regresar todo
    return X_train_raw, X_val_raw, X_test_raw, X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


# Tunear familia de modelos
@task(name="Tune Model Family")
def tune_model_family(X_train_proc, X_val_proc, y_train, y_val, model_family: str, n_trials: int = 10, random_state: int = 42):
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.trial.Trial):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", model_family)

            if model_family == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "random_state": random_state,
                    "n_jobs": -1
                }
                model = RandomForestClassifier(**params)

            elif model_family == "gradient_boosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "random_state": random_state
                }
                model = GradientBoostingClassifier(**params)

            elif model_family == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    "random_state": random_state,
                    "n_jobs": -1,
                    "use_label_encoder": False,
                    "eval_metric": "logloss"
                }
                model = XGBClassifier(**params)

            else:
                raise ValueError(f"Unsupported model family: {model_family}")

            mlflow.log_params(params)

            model.fit(X_train_proc, y_train)
            y_pred = model.predict(X_val_proc)

            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)

            mlflow.log_metrics({"f1": f1, "precision": precision, "recall": recall})

            # log modelo candidato (sin preprocesador) para que la corrida de hiperparámetros tenga el artefacto candidato
            input_example = pd.DataFrame(X_val_proc.head(5), columns=X_val_proc.columns)
            try:
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=input_example)
            except Exception:
                pass

        # Optuna minimiza por eso el return es 1 - f1
        return 1.0 - f1

    with mlflow.start_run(run_name=f"{model_family.title()} Hyperparameter Optimization (Optuna)"):
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params.copy()
        best_params["random_state"] = random_state
        if model_family == "random_forest":
            best_params["n_jobs"] = -1

        return best_params


# Entrenar mdelo final candidato
@task(name="Train Final Challenger")
def train_final_challenger(X_train_raw, X_val_raw, X_train_proc, X_val_proc, y_train, y_val, preprocessor, best_params: dict, model_family: str):
    """Entrenar modelo final candidato (preprocessor + model) para que el modelo registrado
       contenga el preprocesador usable por la API."""
    with mlflow.start_run(run_name=f"Challenger Modelo: {model_family.title()}") as run:
        mlflow.set_tag("model_family", model_family)
        mlflow.log_params(best_params)

        if model_family == "random_forest":
            model = RandomForestClassifier(**best_params)
        elif model_family == "gradient_boosting":
            model = GradientBoostingClassifier(**best_params)
        elif model_family == "xgboost":
            bp = best_params.copy()
            bp.setdefault("use_label_encoder", False)
            bp.setdefault("eval_metric", "logloss")
            model = XGBClassifier(**bp)
        else:
            raise ValueError(f"Unsupported model family: {model_family}")

        # entrenar con el set completo (train + val)
        model.fit(X_train_proc, y_train)

        y_pred = model.predict(X_val_proc)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)

        mlflow.log_metrics({"f1": f1, "precision": precision, "recall": recall})

        # Construir un Pipeline completo de sklearn: el preprocesador espera un DataFrame raw como input
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Preparar input example (filas raw de X_val) y signature
        input_example = X_val_raw.head(5)
        signature = infer_signature(input_example, y_val.head(5))

        # Log el pipeline completo
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        return run.info.run_id


# Comparar y promover champion
@task(name="Compare and Promote Champion")
def compare_and_promote(experiment_id: str):
    client = MlflowClient()

    # encontrar runs challenger ordenados por f1 desc (mejor primero)
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.model_family IN ('random_forest','gradient_boosting','xgboost') AND attributes.run_name LIKE 'Challenger%'",
        order_by=["metrics.f1 DESC"],
    )

    top_runs = runs_df.head(2)

    # Crear modelo registrado si no existe
    try:
        client.get_registered_model(name=MODEL_NAME_UC)
    except Exception:
        try:
            client.create_registered_model(name=MODEL_NAME_UC)
        except Exception as ee:
            print(f"Failed creating registered model: {ee}")

    for idx, (_, row) in enumerate(top_runs.iterrows()):
        run_id = row["run_id"]
        model_uri = f"runs:/{run_id}/model"

        # crear versión de modelo o encontrar versión existente para esa corrida
        try:
            mv = client.create_model_version(
                name=MODEL_NAME_UC,
                source=model_uri,
                run_id=run_id
            )
        except Exception:
            mv = None
            versions = client.search_model_versions(f"name = '{MODEL_NAME_UC}'")
            for v in versions:
                if getattr(v, "run_id", None) == run_id or getattr(v, "run_id", None) == row["run_id"]:
                    mv = v
                    break
            if mv is None:
                print(f"No model version found for run {run_id}, skipping.")
                continue

        alias = "champion" if idx == 0 else "challenger"

        try:
            client.set_registered_model_alias(
                name=MODEL_NAME_UC,
                alias=alias,
                version=mv.version
            )
        except Exception as e:
            print(f"Error setting alias {alias} => {e}")


# Flujo principal
@flow(name="Income_Prediction")
def income_challenger_flow(file_path: str, n_trials_per_family: int = 10):
    load_dotenv(override=True)

    # asegurar que el experimento exista y obtener su id
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    # cargar datos raw y procesados
    results = load_and_preprocess(file_path)
    (X_train_raw, X_val_raw, X_test_raw,
     X_train_proc, X_val_proc, X_test_proc,
     y_train, y_val, y_test,
     preprocessor) = results

    # ajustar familias
    best_rf = tune_model_family(X_train_proc, X_val_proc, y_train, y_val, "random_forest", n_trials=n_trials_per_family)
    best_gb = tune_model_family(X_train_proc, X_val_proc, y_train, y_val, "gradient_boosting", n_trials=n_trials_per_family)
    best_xgb = tune_model_family(X_train_proc, X_val_proc, y_train, y_val, "xgboost", n_trials=n_trials_per_family)

    # entrenar challengers finales y loguear pipeline completo (preprocesador + modelo)
    run_id_rf = train_final_challenger(X_train_raw, X_val_raw, X_train_proc, X_val_proc, y_train, y_val, preprocessor, best_rf, "random_forest")
    run_id_gb = train_final_challenger(X_train_raw, X_val_raw, X_train_proc, X_val_proc, y_train, y_val, preprocessor, best_gb, "gradient_boosting")
    run_id_xgb = train_final_challenger(X_train_raw, X_val_raw, X_train_proc, X_val_proc, y_train, y_val, preprocessor, best_xgb, "xgboost")

    # promover champion
    compare_and_promote(experiment_id)

    print("Pipeline finalizado.")


if __name__ == "__main__":
    
    data_file_path = "../../data/raw/adult.csv"
    income_challenger_flow(file_path=data_file_path, n_trials_per_family=10)