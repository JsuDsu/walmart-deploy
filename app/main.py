from fastapi import Depends, FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from dotenv import load_dotenv
from .models import SalesInput, SalesOutput, ChatRequest, TokenResponse, UserCreate, UserPublic
from ml_model import load_or_train_model, get_metrics
from .auth_db import init_auth_schema, create_user, list_users, get_user_by_username
from .security import (
    ROLE_LABELS,
    create_access_token,
    permissions_for_role,
    verify_password,
)
from .deps import get_current_user, require_permission
from .reports import (
    build_dashboard_report,
    export_sales_csv,
    get_dataset_date_range,
    init_reports_schema,
    log_prediction,
    validate_date_range,
)
from .pdf_reports import generate_reports_pdf
import io
import logging
import pandas as pd
import numpy as np
import os
from datetime import date
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE)

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def _gemini_ready() -> bool:
    if not genai or not GEMINI_API_KEY:
        return False
    placeholder = GEMINI_API_KEY.strip().lower()
    return placeholder not in ("", "tu_api_key_aqui", "your_api_key_here")


if _gemini_ready():
    genai.configure(api_key=GEMINI_API_KEY)
elif not GEMINI_API_KEY:
    logging.getLogger(__name__).warning(
        "GEMINI_API_KEY no encontrada. Crea %s con tu clave de Google AI Studio.",
        _ENV_FILE,
    )

CHAT_SYSTEM_PROMPT = (
    "Eres un asistente experto en la aplicación Walmart Sales Predictor. "
    "Solo debes responder preguntas relacionadas con esta aplicación, su uso, su configuración, "
    "sus rutas y cómo funciona. "
    "Si la pregunta no es sobre la aplicación, responde: "
    "'Lo siento, solo puedo responder preguntas sobre la aplicación Walmart Sales Predictor.'"
)

app = FastAPI(
    title="Walmart Sales Prediction API",
    description="Predicción de ventas semanales con Random Forest y validación temporal",
    version="2.0.0"
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cargar modelo al iniciar
MODEL_NAME = "RandomForest"
model, scaler, metrics = load_or_train_model(model_name=MODEL_NAME)

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


@app.on_event("startup")
def startup_init_auth():
    try:
        init_auth_schema()
        init_reports_schema()
    except Exception as exc:
        logger.warning("No se pudo inicializar tablas en MySQL: %s", exc)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/auth/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_username(form_data.username)
    if not user or not user.get("is_active"):
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")
    if not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")

    role = user["role"]
    token = create_access_token(user["username"], role)
    return TokenResponse(
        access_token=token,
        username=user["username"],
        role=role,
        role_label=ROLE_LABELS.get(role, role),
        permissions=permissions_for_role(role),
    )


@app.get("/auth/me")
def auth_me(current_user: dict = Depends(get_current_user)):
    role = current_user["role"]
    return {
        "username": current_user["username"],
        "role": role,
        "role_label": ROLE_LABELS.get(role, role),
        "permissions": current_user["permissions"],
    }


@app.get("/auth/users", response_model=list[UserPublic])
def auth_list_users(_: dict = Depends(require_permission("users"))):
    return list_users()


@app.post("/auth/users", response_model=UserPublic, status_code=201)
def auth_create_user(
    payload: UserCreate,
    _: dict = Depends(require_permission("users")),
):
    try:
        created = create_user(payload.username, payload.password, payload.role)
        return UserPublic(id=created["id"], username=created["username"], role=created["role"], is_active=1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        from mysql.connector.errors import IntegrityError
        if isinstance(exc, IntegrityError) or "1062" in str(exc):
            raise HTTPException(status_code=400, detail="El nombre de usuario ya existe") from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "OK", "model_loaded": True, "model_name": metrics.get("model", "Unknown")}


@app.get("/metrics")
def get_model_metrics(_: dict = Depends(require_permission("metrics"))):
    return get_metrics()


@app.post("/predict", response_model=SalesOutput)
def predict(
    data: SalesInput,
    current_user: dict = Depends(require_permission("predict")),
):
    try:
        input_df = pd.DataFrame([data.dict()])[FEATURES]
        input_df[SCALE_COLS] = scaler.transform(input_df[SCALE_COLS])
        log_pred = model.predict(input_df)[0]

        if log_pred > 30:
            log_pred = 30.0
        elif log_pred < 10:
            log_pred = 10.0

        weekly_sales = float(np.expm1(log_pred))
        log_prediction(current_user["username"], data.dict(), float(log_pred), weekly_sales)
        return SalesOutput(log_Weekly_Sales=log_pred, Weekly_Sales=weekly_sales)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/reports/date-range")
def reports_date_range(_: dict = Depends(require_permission("reports"))):
    try:
        return get_dataset_date_range()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/reports/dashboard")
def reports_dashboard(
    fecha_desde: date | None = Query(None, description="Fecha inicial (YYYY-MM-DD)"),
    fecha_hasta: date | None = Query(None, description="Fecha final (YYYY-MM-DD)"),
    _: dict = Depends(require_permission("reports")),
):
    try:
        validate_date_range(fecha_desde, fecha_hasta)
        return build_dashboard_report(fecha_desde, fecha_hasta)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error generando reportes")
        raise HTTPException(status_code=500, detail=f"No se pudieron generar reportes: {exc}") from exc


@app.get("/reports/export")
def reports_export(
    fecha_desde: date | None = Query(None),
    fecha_hasta: date | None = Query(None),
    _: dict = Depends(require_permission("reports")),
):
    try:
        validate_date_range(fecha_desde, fecha_hasta)
        csv_content = export_sales_csv(fecha_desde, fecha_hasta)
        buffer = io.StringIO(csv_content)
        suffix = ""
        if fecha_desde and fecha_hasta:
            suffix = f"_{fecha_desde}_{fecha_hasta}"
        elif fecha_desde:
            suffix = f"_desde_{fecha_desde}"
        elif fecha_hasta:
            suffix = f"_hasta_{fecha_hasta}"
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=reporte_ventas_walmart{suffix}.csv"
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/reports/export-pdf")
def reports_export_pdf(
    fecha_desde: date | None = Query(None),
    fecha_hasta: date | None = Query(None),
    current_user: dict = Depends(require_permission("reports")),
):
    try:
        validate_date_range(fecha_desde, fecha_hasta)
        pdf_bytes = generate_reports_pdf(
            current_user["username"],
            fecha_desde=fecha_desde,
            fecha_hasta=fecha_hasta,
        )
        suffix = ""
        if fecha_desde and fecha_hasta:
            suffix = f"_{fecha_desde}_{fecha_hasta}"
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=reporte_ventas_walmart{suffix}.pdf"
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error generando PDF")
        raise HTTPException(status_code=500, detail=f"No se pudo generar el PDF: {exc}") from exc


@app.post("/retrain")
def retrain_model(
    model_name: str = Query("RandomForest", description="Modelo a entrenar"),
    use_grid_search: bool = Query(False, description="Realizar búsqueda de hiperparámetros"),
    _: dict = Depends(require_permission("retrain")),
):
    global model, scaler, metrics
    try:
        grid_params = None
        if use_grid_search and model_name == "RandomForest":
            grid_params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif use_grid_search and model_name == "GradientBoosting":
            grid_params = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        model, scaler, metrics = load_or_train_model(
            model_name=model_name,
            force_retrain=True,
            use_grid_search=use_grid_search,
            grid_params=grid_params
        )
        return {"message": f"Modelo {model_name} reentrenado exitosamente", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def query_gemini_chat(message: str) -> str | None:
    if not _gemini_ready():
        return None

    models_to_try = [GEMINI_MODEL, "gemini-2.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.0-flash-lite"]
    seen = set()
    for model_name in models_to_try:
        if model_name in seen:
            continue
        seen.add(model_name)
        try:
            gemini = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=CHAT_SYSTEM_PROMPT,
            )
            response = gemini.generate_content(
                message,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 400,
                },
            )
            if response.text:
                return response.text.strip()
        except Exception as exc:
            logger.warning("Gemini API error (%s): %s", model_name, exc)
    return None


def get_chat_answer(message: str) -> str:
    text = message.strip().lower()
    if not text:
        return "Escribe tu pregunta sobre la aplicación o su uso."

    unrelated_triggers = [
        "revolución francesa", "historia", "guerra", "fútbol", "política", "ciencia", "tecnología general",
        "clima", "música", "película", "serie", "economía mundial"
    ]
    if any(term in text for term in unrelated_triggers):
        return "Lo siento, solo puedo responder preguntas sobre la aplicación Walmart Sales Predictor."

    if not _gemini_ready():
        return (
            "El chatbot con Gemini no está configurado. Crea el archivo `.env` en la carpeta "
            "`walmart_sales_api` con esta línea (usa tu clave real):\n\n"
            "GEMINI_API_KEY=AIza...\n\n"
            "Obtén una clave gratis en https://aistudio.google.com/apikey "
            "y reinicia el servidor (`uvicorn app.main:app --reload`)."
        )

    ai_answer = query_gemini_chat(message)
    if ai_answer:
        return ai_answer

    # Fallback básico si falla la API de Gemini.
    if any(term in text for term in ["empresa", "negocio", "beneficio", "ayuda a mi"]):
        return (
            "Esta aplicación ayuda a planificar ventas semanales con datos históricos y variables "
            "como temperatura, CPI y desempleo. Permite estimar ingresos por tienda, apoyar inventario "
            "y decisiones comerciales sin depender solo de intuición."
        )
    if any(term in text for term in ["para qué", "que hace", "funciona", "sirve", "propósito"]):
        return (
            "Esta aplicación predice ventas semanales de Walmart usando un modelo de machine learning. "
            "Carga datos de MySQL, entrena un modelo y devuelve una estimación de ventas en USD."
        )
    if any(term in text for term in ["phpmyadmin", "mysql", "base de datos", "db_config", "walmart_db"]):
        return (
            "La app usa MySQL para cargar los datos de entrenamiento. "
            "Debes configurar `app/db_config.py` con tu host, usuario y contraseña de MySQL. "
            "En XAMPP normalmente es `localhost`, usuario `root` y contraseña vacía si phpMyAdmin no pide credenciales."
        )
    if any(term in text for term in ["entrenar", "reentrenar", "train", "modelo", "gridsearch"]):
        return (
            "Puedes reentrenar el modelo usando la ruta `POST /retrain` o el botón 'Reentrenar modelo' en la web. "
            "La función usa la tabla `sales_training` en MySQL y guarda el modelo en `models/walmart_model.pkl`."
        )
    if any(term in text for term in ["predecir", "predict", "ventas", "input", "datos"]):
        return (
            "Para predecir envía 10 características: ventas promedio, tamaño de tienda, semana, mes, año, fin de año, feriado, temperatura, CPI y desempleo. "
            "La ruta es `POST /predict` y devuelve `Weekly_Sales` en dólares y `log_Weekly_Sales`."
        )
    if any(term in text for term in ["ruta", "endpoint", "health", "metrics", "home"]):
        return (
            "La app ofrece estas rutas principales: `/` para la web, `/predict` para predecir ventas, `/metrics` para métricas, `/retrain` para reentrenar y `/chat` para preguntas."
        )
    return (
        "Puedo ayudarte con preguntas sobre esta app: qué hace, cómo predecir ventas, cómo conectar a MySQL/phpMyAdmin, "
        "y cómo reentrenar el modelo. Prueba a preguntar algo como '¿Qué hace la app?' o '¿Cómo me conecto a phpMyAdmin?'."
    )


@app.post("/chat")
def chat(request: ChatRequest, _: dict = Depends(require_permission("chat"))):
    return {"answer": get_chat_answer(request.message)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)