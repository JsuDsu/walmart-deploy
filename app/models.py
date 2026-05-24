from pydantic import BaseModel, Field

class SalesInput(BaseModel):
    Store_Avg_Sales: float = Field(..., description="Ventas promedio históricas de la tienda (USD)")
    Store_Size_Tier_Enc: int = Field(..., ge=0, le=2, description="0=Baja, 1=Media, 2=Alta")
    Week: int = Field(..., ge=1, le=53)
    Month: int = Field(..., ge=1, le=12)
    Year: int = Field(..., description="2010, 2011 o 2012")
    Is_EndOfYear: int = Field(..., ge=0, le=1, description="1 si mes es Nov o Dic")
    Holiday_Flag: int = Field(..., ge=0, le=1, description="1 si es semana festiva")
    Temperature: float = Field(..., description="Temperatura (°F)")
    CPI: float = Field(..., description="Índice de Precios al Consumidor")
    Unemployment: float = Field(..., description="Tasa de desempleo (%)")

    class Config:
        json_schema_extra = {
            "example": {
                "Store_Avg_Sales": 1555264.40,
                "Store_Size_Tier_Enc": 2,
                "Week": 10,
                "Month": 3,
                "Year": 2011,
                "Is_EndOfYear": 0,
                "Holiday_Flag": 0,
                "Temperature": 57.79,
                "CPI": 214.1110564,
                "Unemployment": 7.742
            }
        }

class SalesOutput(BaseModel):
    log_Weekly_Sales: float = Field(..., description="Ventas transformadas (log1p)")
    Weekly_Sales: float = Field(..., description="Ventas originales en USD")

class TrainRequest(BaseModel):
    model_name: str = Field("RandomForest", description="Modelo a entrenar: LinearRegression, RandomForest, GradientBoosting")
    use_grid_search: bool = Field(False, description="Si se debe realizar búsqueda de hiperparámetros")

class ChatRequest(BaseModel):
    message: str = Field(..., description="Mensaje del usuario para el chatbot")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
    role_label: str
    permissions: list[str]


class UserPublic(BaseModel):
    id: int
    username: str
    role: str
    is_active: int


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    role: str = Field(..., pattern="^(admin|analyst|viewer)$")