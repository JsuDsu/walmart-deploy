"""Generación de reportes PDF con gráficas por año."""
from io import BytesIO
from datetime import date, datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .reports import MONTH_NAMES, STORE_SIZE_LABELS, build_dashboard_report, load_sales_dataframe

WALMART_BLUE = "#0071CE"
CHART_DPI = 120
CHART_WIDTH = 6.5
CHART_HEIGHT = 3.2


def _fig_to_image(fig, width: float = CHART_WIDTH) -> Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    height = width * (CHART_HEIGHT / CHART_WIDTH)
    return Image(buf, width=width * inch, height=height * inch)


def _format_usd(value: float) -> str:
    return f"${value:,.0f}"


def _chart_year_totals(df: pd.DataFrame) -> Image:
    by_year = df.groupby("Year")["Weekly_Sales"].sum()
    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    years = [str(int(y)) for y in by_year.index]
    vals = by_year.values / 1e6
    bars = ax.bar(years, vals, color=WALMART_BLUE, edgecolor="white")
    ax.set_title("Ventas totales por año (millones USD)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Millones USD")
    ax.bar_label(bars, fmt="%.1fM", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _fig_to_image(fig)


def _chart_monthly_by_year(df: pd.DataFrame, year: int) -> Image:
    subset = df[df["Year"] == year]
    monthly = subset.groupby("Month")["Weekly_Sales"].mean()
    months = list(range(1, 13))
    vals = [monthly.get(m, np.nan) for m in months]
    labels = [MONTH_NAMES[m] for m in months]

    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    ax.plot(labels, vals, marker="o", color=WALMART_BLUE, linewidth=2, markersize=6)
    ax.fill_between(range(len(labels)), vals, alpha=0.15, color=WALMART_BLUE)
    ax.set_title(f"Promedio mensual de ventas — {year}", fontsize=12, fontweight="bold")
    ax.set_ylabel("USD")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
    ax.grid(alpha=0.3)
    return _fig_to_image(fig)


def _chart_store_size(df: pd.DataFrame) -> Image:
    by_store = df.groupby("Store_Size_Tier_Enc")["Weekly_Sales"].mean()
    labels = [STORE_SIZE_LABELS.get(int(k), str(k)) for k in by_store.index]
    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    colors_pie = ["#94a3b8", WALMART_BLUE, "#003f6b"]
    ax.pie(by_store.values, labels=labels, autopct="%1.1f%%", colors=colors_pie[: len(labels)], startangle=90)
    ax.set_title("Distribución del promedio de ventas por tamaño de tienda", fontsize=12, fontweight="bold")
    return _fig_to_image(fig)


def _chart_holiday(df: pd.DataFrame) -> Image:
    holiday = df.groupby("Holiday_Flag")["Weekly_Sales"].mean()
    labels = ["Semana normal", "Semana festiva"]
    vals = [holiday.get(0, 0), holiday.get(1, 0)]
    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    bars = ax.bar(labels, vals, color=["#22c55e", "#f59e0b"], edgecolor="white")
    ax.set_title("Promedio semanal: festivos vs normales", fontsize=12, fontweight="bold")
    ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
    ax.bar_label(bars, labels=[_format_usd(v) for v in vals], fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _fig_to_image(fig)


def _chart_end_of_year(df: pd.DataFrame) -> Image:
    eoy = df.groupby("Is_EndOfYear")["Weekly_Sales"].mean()
    labels = ["Resto del año", "Fin de año (Nov-Dic)"]
    vals = [eoy.get(0, 0), eoy.get(1, 0)]
    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    bars = ax.bar(labels, vals, color=["#6366f1", "#ec4899"], edgecolor="white")
    ax.set_title("Promedio semanal: fin de año vs resto", fontsize=12, fontweight="bold")
    ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
    ax.bar_label(bars, labels=[_format_usd(v) for v in vals], fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _fig_to_image(fig)


def _format_period(fecha_desde: date | None, fecha_hasta: date | None, report: dict) -> str:
    dr = report.get("date_range") or {}
    if dr.get("applied_desde") and dr.get("applied_hasta"):
        return f"{dr['applied_desde']} al {dr['applied_hasta']}"
    if fecha_desde and fecha_hasta:
        return f"{fecha_desde} al {fecha_hasta}"
    return "Todos los datos disponibles"


def generate_reports_pdf(
    username: str = "",
    fecha_desde: date | None = None,
    fecha_hasta: date | None = None,
) -> bytes:
    report = build_dashboard_report(fecha_desde, fecha_hasta)
    period_label = _format_period(fecha_desde, fecha_hasta, report)
    generated = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    if report.get("empty"):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("Reporte de ventas Walmart", styles["Heading1"]),
            Spacer(1, 0.3 * inch),
            Paragraph(f"Período solicitado: {period_label}", styles["Normal"]),
            Spacer(1, 0.2 * inch),
            Paragraph(report.get("message", "Sin datos en el rango seleccionado."), styles["Normal"]),
        ]
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    df = load_sales_dataframe(fecha_desde, fecha_hasta)
    summary = report["summary"]

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor(WALMART_BLUE),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#003f6b"),
        spaceBefore=16,
        spaceAfter=8,
    )

    story = []
    story.append(Paragraph("Reporte de ventas Walmart", title_style))
    user_line = f"Generado por: <b>{username}</b> · " if username else ""
    story.append(Paragraph(f"{user_line}Fecha: {generated}", subtitle_style))
    story.append(Paragraph(f"Período del reporte: <b>{period_label}</b>", subtitle_style))
    story.append(
        Paragraph(
            f"Registros incluidos: <b>{report['date_range']['records_filtered']:,}</b>",
            subtitle_style,
        )
    )

    kpi_data = [
        ["Indicador", "Valor"],
        ["Registros analizados", f"{summary['total_records']:,}"],
        ["Ventas totales", _format_usd(summary["total_sales_usd"])],
        ["Promedio semanal", _format_usd(summary["avg_weekly_sales"])],
        ["Máximo semanal", _format_usd(summary["max_weekly_sales"])],
        ["Temperatura promedio", f"{summary['avg_temperature']} °F"],
        ["Desempleo promedio", f"{summary['avg_unemployment']}%"],
    ]
    kpi_table = Table(kpi_data, colWidths=[3 * inch, 3.5 * inch])
    kpi_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(WALMART_BLUE)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ])
    )
    story.append(kpi_table)
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Resumen por año", heading_style))
    year_table_data = [["Año", "Ventas totales", "Promedio semanal", "Registros"]]
    for row in report["by_year"]:
        year_table_data.append([
            str(row["year"]),
            _format_usd(row["total_sales"]),
            _format_usd(row["avg_sales"]),
            str(row["records"]),
        ])
    yt = Table(year_table_data, colWidths=[1 * inch, 1.8 * inch, 1.8 * inch, 1.2 * inch])
    yt.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003f6b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )
    story.append(yt)
    story.append(Spacer(1, 0.2 * inch))
    story.append(_chart_year_totals(df))

    years = sorted(df["Year"].unique())
    story.append(PageBreak())
    story.append(Paragraph("Gráficas mensuales por año (período filtrado)", heading_style))
    for year in years:
        story.append(Paragraph(f"Año {int(year)}", styles["Heading3"]))
        story.append(Spacer(1, 0.1 * inch))
        story.append(_chart_monthly_by_year(df, int(year)))
        story.append(Spacer(1, 0.15 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Análisis comparativo", heading_style))
    story.append(_chart_store_size(df))
    story.append(Spacer(1, 0.15 * inch))
    story.append(_chart_holiday(df))
    story.append(Spacer(1, 0.15 * inch))
    story.append(_chart_end_of_year(df))

    story.append(PageBreak())
    story.append(Paragraph("Top 10 semanas con mayores ventas", heading_style))
    top_data = [["Año", "Mes", "Sem.", "Tienda", "Ventas (USD)"]]
    for w in report["top_weeks"]:
        top_data.append([
            str(w["Year"]),
            str(w["Month"]),
            str(w["Week"]),
            w["store_size"],
            _format_usd(w["Weekly_Sales"]),
        ])
    top_table = Table(top_data, colWidths=[0.7 * inch, 0.7 * inch, 0.7 * inch, 1.2 * inch, 1.8 * inch])
    top_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(WALMART_BLUE)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )
    story.append(top_table)

    metrics = report.get("model_metrics") or {}
    if metrics:
        story.append(Spacer(1, 0.25 * inch))
        story.append(Paragraph("Métricas del modelo ML", heading_style))
        model_text = (
            f"Modelo: <b>{metrics.get('model', 'N/A')}</b> · "
            f"R² CV: <b>{metrics.get('cv_r2_mean', 0):.4f}</b> · "
            f"MAE: <b>{metrics.get('cv_mae_mean', 0):.4f}</b> · "
            f"RMSE: <b>{metrics.get('cv_rmse_mean', 0):.4f}</b>"
        )
        story.append(Paragraph(model_text, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
