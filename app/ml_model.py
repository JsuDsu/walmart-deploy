import os
import joblib
import json
from training import train_model

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "walmart_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

def load_or_train_model(model_name="RandomForest", force_retrain=False, use_grid_search=False, grid_params=None):
    """Carga el modelo existente o lo entrena si no existe o se fuerza retrain."""
    if not force_retrain and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        metrics = {}
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
        print("Modelo cargado desde disco.")
        return model, scaler, metrics
    else:
        print("Entrenando nuevo modelo...")
        model, scaler, metrics = train_model(model_name, use_grid_search, grid_params)
        return model, scaler, metrics

def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {}