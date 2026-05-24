import logging
from datetime import date, datetime, timezone

import mysql.connector
import numpy as np
import pandas as pd

from .db_config import DB_CONFIG
from .ml_model import get_metrics

logger = logging.getLogger(__name__)

STORE_SIZE_LABELS = {0: "Baja", 1: "Media", 2: "Alta"}
MONTH_NAMES = [
    "", "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
]


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def week_to_date(year: int, week: int) -> date:
    week = max(1, min(int(week), 52))
    try:
        return date.fromisocalendar(int(year), week, 1)
    except ValueError:
        return date(int(year), 1, 1)


def add_report_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["report_date"] = out.apply(
        lambda r: week_to_date(r["Year"], r["Week"]),
        axis=1,
    )
    return out


def validate_date_range(fecha_desde: date | None, fecha_hasta: date | None) -> None:
    if fecha_desde and fecha_hasta and fecha_desde > fecha_hasta:
        raise ValueError("La fecha inicial no puede ser posterior a la fecha final")


def get_dataset_date_range() -> dict:
    df = add_report_date(_load_sales_raw())
    return {
        "min_date": df["report_date"].min().isoformat(),
        "max_date": df["report_date"].max().isoformat(),
    }


def _load_sales_raw() -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql(
            """
            SELECT Store_Avg_Sales, Store_Size_Tier_Enc, Week, Month, Year,
                   Is_EndOfYear, Holiday_Flag, Temperature, CPI, Unemployment,
                   log_Weekly_Sales
            FROM sales_training
            """,
            conn,
        )
    finally:
        if conn.is_connected():
            conn.close()
    df["Weekly_Sales"] = np.expm1(df["log_Weekly_Sales"])
    return df


def init_reports_schema() -> None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL,
                Store_Avg_Sales FLOAT,
                Store_Size_Tier_Enc INT,
                Week INT,
                Month INT,
                Year INT,
                Is_EndOfYear INT,
                Holiday_Flag INT,
                Temperature FLOAT,
                CPI FLOAT,
                Unemployment FLOAT,
                log_Weekly_Sales FLOAT,
                Weekly_Sales FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        if conn.is_connected():
            conn.close()


def load_sales_dataframe(
    fecha_desde: date | None = None,
    fecha_hasta: date | None = None,
) -> pd.DataFrame:
    validate_date_range(fecha_desde, fecha_hasta)
    df = add_report_date(_load_sales_raw())
    if fecha_desde is not None:
        df = df[df["report_date"] >= fecha_desde]
    if fecha_hasta is not None:
        df = df[df["report_date"] <= fecha_hasta]
    return df


def log_prediction(username: str, payload: dict, log_sales: float, weekly_sales: float) -> None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_logs (
                username, Store_Avg_Sales, Store_Size_Tier_Enc, Week, Month, Year,
                Is_EndOfYear, Holiday_Flag, Temperature, CPI, Unemployment,
                log_Weekly_Sales, Weekly_Sales
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                username,
                payload["Store_Avg_Sales"],
                payload["Store_Size_Tier_Enc"],
                payload["Week"],
                payload["Month"],
                payload["Year"],
                payload["Is_EndOfYear"],
                payload["Holiday_Flag"],
                payload["Temperature"],
                payload["CPI"],
                payload["Unemployment"],
                log_sales,
                weekly_sales,
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.warning("No se pudo guardar predicción en reportes: %s", exc)
    finally:
        if conn.is_connected():
            conn.close()


def get_recent_predictions(
    limit: int = 15,
    fecha_desde: date | None = None,
    fecha_hasta: date | None = None,
) -> list[dict]:
    conn = get_connection()
    try:
        query = """
            SELECT id, username, Week, Month, Year, Store_Size_Tier_Enc,
                   Holiday_Flag, Weekly_Sales, created_at
            FROM prediction_logs
            WHERE 1=1
        """
        params: list = []
        if fecha_desde is not None:
            query += " AND DATE(created_at) >= %s"
            params.append(fecha_desde.isoformat())
        if fecha_hasta is not None:
            query += " AND DATE(created_at) <= %s"
            params.append(fecha_hasta.isoformat())
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        df = pd.read_sql(query, conn, params=tuple(params))
    except Exception:
        return []
    finally:
        if conn.is_connected():
            conn.close()

    rows = []
    for _, row in df.iterrows():
        created = row["created_at"]
        rows.append({
            "id": int(row["id"]),
            "username": row["username"],
            "week": int(row["Week"]),
            "month": int(row["Month"]),
            "year": int(row["Year"]),
            "store_size": STORE_SIZE_LABELS.get(int(row["Store_Size_Tier_Enc"]), str(row["Store_Size_Tier_Enc"])),
            "holiday": bool(row["Holiday_Flag"]),
            "weekly_sales": round(float(row["Weekly_Sales"]), 2),
            "created_at": created.isoformat() if hasattr(created, "isoformat") else str(created),
        })
    return rows


def build_dashboard_report(
    fecha_desde: date | None = None,
    fecha_hasta: date | None = None,
) -> dict:
    validate_date_range(fecha_desde, fecha_hasta)
    df = load_sales_dataframe(fecha_desde, fecha_hasta)
    model_metrics = get_metrics()

    if df.empty:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "empty": True,
            "message": "No hay datos en el rango de fechas seleccionado.",
            "date_range": {
                "fecha_desde": fecha_desde.isoformat() if fecha_desde else None,
                "fecha_hasta": fecha_hasta.isoformat() if fecha_hasta else None,
                "records_filtered": 0,
            },
            "summary": {},
            "by_year": [],
            "by_month": [],
            "by_store_size": [],
            "holiday_comparison": [],
            "end_of_year_comparison": [],
            "top_weeks": [],
            "temperature_impact": [],
            "model_metrics": model_metrics,
            "recent_predictions": get_recent_predictions(
                fecha_desde=fecha_desde,
                fecha_hasta=fecha_hasta,
            ),
        }

    by_year = (
        df.groupby("Year")["Weekly_Sales"]
        .agg(total="sum", promedio="mean", registros="count")
        .reset_index()
    )
    by_year_list = [
        {
            "year": int(r.Year),
            "total_sales": round(float(r.total), 2),
            "avg_sales": round(float(r.promedio), 2),
            "records": int(r.registros),
        }
        for r in by_year.itertuples()
    ]

    by_month = (
        df.groupby(["Year", "Month"])["Weekly_Sales"]
        .mean()
        .reset_index()
        .sort_values(["Year", "Month"])
    )
    by_month_list = [
        {
            "year": int(r.Year),
            "month": int(r.Month),
            "month_label": MONTH_NAMES[int(r.Month)],
            "avg_sales": round(float(r.Weekly_Sales), 2),
        }
        for r in by_month.itertuples()
    ]

    by_store = df.groupby("Store_Size_Tier_Enc")["Weekly_Sales"].mean().reset_index()
    by_store_list = [
        {
            "tier": int(r.Store_Size_Tier_Enc),
            "label": STORE_SIZE_LABELS.get(int(r.Store_Size_Tier_Enc), "N/A"),
            "avg_sales": round(float(r.Weekly_Sales), 2),
            "count": int((df["Store_Size_Tier_Enc"] == r.Store_Size_Tier_Enc).sum()),
        }
        for r in by_store.itertuples()
    ]

    holiday = df.groupby("Holiday_Flag")["Weekly_Sales"].mean()
    holiday_list = [
        {
            "holiday": bool(int(flag)),
            "label": "Semana festiva" if int(flag) else "Semana normal",
            "avg_sales": round(float(holiday.loc[flag]), 2),
        }
        for flag in holiday.index
    ]

    end_year = df.groupby("Is_EndOfYear")["Weekly_Sales"].mean()
    end_year_list = [
        {
            "end_of_year": bool(int(flag)),
            "label": "Fin de año (Nov-Dic)" if int(flag) else "Resto del año",
            "avg_sales": round(float(end_year.loc[flag]), 2),
        }
        for flag in end_year.index
    ]

    top_weeks = (
        df.nlargest(10, "Weekly_Sales")[
            ["Year", "Month", "Week", "Weekly_Sales", "Store_Size_Tier_Enc", "Holiday_Flag"]
        ]
        .to_dict(orient="records")
    )
    for row in top_weeks:
        row["Weekly_Sales"] = round(float(row["Weekly_Sales"]), 2)
        row["store_size"] = STORE_SIZE_LABELS.get(int(row.pop("Store_Size_Tier_Enc")), "N/A")
        row["holiday"] = bool(row.pop("Holiday_Flag"))

    temp_bins = pd.cut(df["Temperature"], bins=5)
    temp_avg = df.groupby(temp_bins, observed=True)["Weekly_Sales"].mean()
    temp_list = [
        {"range": str(interval), "avg_sales": round(float(val), 2)}
        for interval, val in temp_avg.items()
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "empty": False,
        "date_range": {
            "fecha_desde": fecha_desde.isoformat() if fecha_desde else None,
            "fecha_hasta": fecha_hasta.isoformat() if fecha_hasta else None,
            "applied_desde": pd.Timestamp(df["report_date"].min()).strftime("%Y-%m-%d"),
            "applied_hasta": pd.Timestamp(df["report_date"].max()).strftime("%Y-%m-%d"),
            "records_filtered": int(len(df)),
        },
        "summary": {
            "total_records": int(len(df)),
            "total_sales_usd": round(float(df["Weekly_Sales"].sum()), 2),
            "avg_weekly_sales": round(float(df["Weekly_Sales"].mean()), 2),
            "min_weekly_sales": round(float(df["Weekly_Sales"].min()), 2),
            "max_weekly_sales": round(float(df["Weekly_Sales"].max()), 2),
            "avg_temperature": round(float(df["Temperature"].mean()), 2),
            "avg_unemployment": round(float(df["Unemployment"].mean()), 2),
        },
        "by_year": by_year_list,
        "by_month": by_month_list,
        "by_store_size": by_store_list,
        "holiday_comparison": holiday_list,
        "end_of_year_comparison": end_year_list,
        "top_weeks": top_weeks,
        "temperature_impact": temp_list,
        "model_metrics": model_metrics,
        "recent_predictions": get_recent_predictions(
            fecha_desde=fecha_desde,
            fecha_hasta=fecha_hasta,
        ),
    }


def export_sales_csv(
    fecha_desde: date | None = None,
    fecha_hasta: date | None = None,
) -> str:
    df = load_sales_dataframe(fecha_desde, fecha_hasta)
    if df.empty:
        return "report_date,Year,Month,Week,store_size_label,Weekly_Sales\n"
    df["store_size_label"] = df["Store_Size_Tier_Enc"].map(STORE_SIZE_LABELS)
    cols = [
        "report_date", "Year", "Month", "Week", "store_size_label", "Weekly_Sales",
        "Holiday_Flag", "Temperature", "CPI", "Unemployment",
    ]
    export_df = df[cols].copy()
    export_df["report_date"] = export_df["report_date"].astype(str)
    return export_df.to_csv(index=False)
