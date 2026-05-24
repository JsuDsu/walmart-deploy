from pydantic import BaseModel, Field

class SalesInput(BaseModel):
    Store_Avg_Sales: float = Field(..., description="Promedio histórico de ventas de la tienda (USD)")
    Store_Size_Tier_Enc: int = Field(..., ge=0, le=2, description="0=Low, 1=Medium, 2=High")
    Week: int = Field(..., ge=1, le=53)
    Month: int = Field(..., ge=1, le=12)
    Year: int = Field(..., ge=2010, le=2012)
    Is_EndOfYear: int = Field(..., ge=0, le=1, description="1 si mes es Nov o Dic")
    Holiday_Flag: int = Field(..., ge=0, le=1, description="1 si semana festiva")
    Temperature: float
    CPI: float
    Unemployment: float

    class Config:
        schema_extra = {
            "example": {
                "Store_Avg_Sales": 1500000.0,
                "Store_Size_Tier_Enc": 2,
                "Week": 10,
                "Month": 3,
                "Year": 2011,
                "Is_EndOfYear": 0,
                "Holiday_Flag": 0,
                "Temperature": 55.0,
                "CPI": 215.0,
                "Unemployment": 7.5
            }
        }

class SalesOutput(BaseModel):
    log_Weekly_Sales: float = Field(..., description="Ventas transformadas (log1p)")
    Weekly_Sales: float = Field(..., description="Ventas originales en USD")