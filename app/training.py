import pandas as pd
import numpy as np
import mysql.connector
import pandas as pd
from .db_config import DB_CONFIG
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json

DATA_DIR = "data"
MODEL_DIR = "models"
FEATURES = [
    "Store_Avg_Sales",
    "Store_Size_Tier_Enc",
    "Week",
    "Month",
    "Year",
    "Is_EndOfYear",
    "Holiday_Flag",
    "Temperature",
    "CPI",
    "Unemployment"
]
SCALE_COLS = ["Temperature", "CPI", "Unemployment", "Store_Avg_Sales"]

def load_data():
    """Carga los datos desde MySQL en lugar de CSV"""
    conn = mysql.connector.connect(**DB_CONFIG)
    query = """
        SELECT Store_Avg_Sales, Store_Size_Tier_Enc, Week, Month, Year,
               Is_EndOfYear, Holiday_Flag, Temperature, CPI, Unemployment,
               log_Weekly_Sales
        FROM sales_training
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    X = df.drop('log_Weekly_Sales', axis=1)
    y = df['log_Weekly_Sales']
    return X, y

def train_model(model_name="RandomForest", use_grid_search=False, grid_params=None):
    X, y = load_data()
    
    # Escalar
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[SCALE_COLS] = scaler.fit_transform(X[SCALE_COLS])
    
    # Validación cruzada temporal (5 splits)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Definir modelos base
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }
    
    if model_name not in models:
        raise ValueError(f"Modelo {model_name} no disponible. Opciones: {list(models.keys())}")
    
    model = models[model_name]
    
    # Para regresión lineal no se realiza búsqueda de hiperparámetros (no tiene parámetros importantes)
    if use_grid_search and model_name != "LinearRegression" and grid_params:
        print(f"Realizando búsqueda de hiperparámetros para {model_name}...")
        grid_search = GridSearchCV(model, grid_params, cv=tscv, scoring='r2', n_jobs=-1)
        grid_search.fit(X_scaled, y)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Mejores parámetros: {best_params}")
    else:
        best_model = model
        best_params = {}
        print(f"Entrenando {model_name} sin búsqueda de hiperparámetros...")
        best_model.fit(X_scaled, y)
    
    # Evaluación con TimeSeriesSplit
    r2_scores = []
    mae_scores = []
    rmse_scores = []
    
    for train_idx, val_idx in tscv.split(X_scaled):
        X_train_fold = X_scaled.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X_scaled.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Crear modelo para el fold respetando la ausencia de random_state en regresión lineal
        if model_name == "LinearRegression":
            fold_model = LinearRegression()
            if best_params:
                fold_model.set_params(**best_params)
        else:
            fold_model = model.__class__(**best_params) if best_params else model.__class__(random_state=42)
        
        fold_model.fit(X_train_fold, y_train_fold)
        y_pred = fold_model.predict(X_val_fold)
        
        r2_scores.append(r2_score(y_val_fold, y_pred))
        mae_scores.append(mean_absolute_error(y_val_fold, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))
    
    metrics = {
        "model": model_name,
        "best_params": best_params,
        "cv_r2_mean": np.mean(r2_scores),
        "cv_r2_std": np.std(r2_scores),
        "cv_mae_mean": np.mean(mae_scores),
        "cv_rmse_mean": np.mean(rmse_scores),
        "train_size": len(X),
        "features": FEATURES
    }
    
    # Guardar modelo y scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "walmart_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    
    # Guardar métricas en JSON
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Modelo {model_name} entrenado y guardado.")
    print(f"Métricas de validación cruzada temporal: R2 = {metrics['cv_r2_mean']:.4f} (+/- {metrics['cv_r2_std']:.4f})")
    return best_model, scaler, metrics