# Walmart Sales Prediction API

## Requisitos previos

- Tener instalado Python 3.10 o superior.
- Tener los archivos `X_train.csv` y `y_train.csv` (generados en la fase anterior) en la carpeta `data/`.

## Instalación y ejecución (paso a paso)

### 1. Clonar / descargar el proyecto

Asegúrate de tener la siguiente estructura de carpetas:
walmart_sales_api/
├── app/
│ ├── init.py
│ ├── main.py
│ ├── models.py
│ ├── ml_model.py
│ └── utils.py
├── data/
│ ├── X_train.csv
│ └── y_train.csv
├── models/ (se creará sola al entrenar)
├── static/
│ └── style.css
├── templates/
│ └── index.html
└── requirements.txt

text

### 2. Instalar dependencias

Abre una terminal en la carpeta `walmart_sales_api` y ejecuta:

```bash
pip install -r requirements.txt
Si no tienes el archivo requirements.txt, créalo con este contenido:

txt
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
joblib==1.3.2
python-multipart==0.0.6
3. Ejecutar la API
En la misma terminal, ejecuta:

bash
uvicorn app.main:app --reload
Verás un mensaje como:

text
INFO:     Uvicorn running on http://127.0.0.1:8000
¡No cierres esta terminal! Déjala corriendo.

4. Usar la API
Abre tu navegador y ve a:
➡️ http://127.0.0.1:8000

Allí encontrarás un formulario para ingresar los datos y obtener la predicción.

También puedes probar con curl en otra terminal:

bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"Store_Avg_Sales\":1555264.40,\"Store_Size_Tier_Enc\":2,\"Week\":10,\"Month\":3,\"Year\":2011,\"Is_EndOfYear\":0,\"Holiday_Flag\":0,\"Temperature\":57.79,\"CPI\":214.1110564,\"Unemployment\":7.742}"
¿Qué hace esta API?
Recibe 10 características de una tienda y una semana, y devuelve las ventas estimadas en dólares. El modelo fue entrenado con datos reales de Walmart (2010-2012).

Solución de problemas
Error ImportError: cannot import name 'SalesOutput'
Asegúrate de que el archivo app/models.py contenga la clase SalesOutput (revisa el código proporcionado).

Error Connection refused
El servidor no está corriendo. Vuelve al paso 3.

Error de archivos CSV faltantes
Coloca X_train.csv y y_train.csv dentro de la carpeta data/.

¿Necesitas ayuda?
Revisa que todos los archivos de código estén en su lugar. Si algo falla, comparte el mensaje de error.

text

Este README es corto, práctico y orientado a que tu compañero lo ejecute sin problemas. Solo debe copiar la estructura, pegar los archivos, instalar dependencias y correr el comando.