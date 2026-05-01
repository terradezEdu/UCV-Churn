from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from math import erf, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from shiny import App, reactive, render, ui


APP_DIR = Path(__file__).resolve().parent
RAW_DIR = APP_DIR / "data" / "raw"
PROCESSED_DIR = APP_DIR / "data" / "processed"

MONTH_ALL = "__all__"
ALL_VALUE = "__all__"

COLOR_CHURN = "#c65d4b"
COLOR_SAFE = "#8091a0"
COLOR_ACCENT = "#0f766e"
COLOR_GRID = "#d7ddd9"
BG_FIG = "#f7f5f0"


def prefer_processed_path(name: str) -> Path:
    processed_path = PROCESSED_DIR / f"{name}.csv"
    return processed_path if processed_path.exists() else RAW_DIR / f"{name}.csv"


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["fecha", "mes", "fecha_evento"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def month_choices(df: pd.DataFrame) -> dict[str, str]:
    choices = {MONTH_ALL: "Todos"}
    for value in (
        df["fecha"].dropna().sort_values().dt.strftime("%Y-%m").drop_duplicates().tolist()
        if "fecha" in df.columns
        else []
    ):
        choices[value] = value
    return choices


def safe_qcut(series: pd.Series, q: int, labels: list[str]) -> pd.Series:
    valid = series.dropna()
    if valid.nunique() < len(labels):
        return pd.Series(pd.NA, index=series.index, dtype="object")
    return pd.qcut(series, q=q, labels=labels, duplicates="drop")


def age_band(series: pd.Series) -> pd.Series:
    bins = [0, 25, 35, 45, 55, 65, 200]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    return pd.cut(series, bins=bins, labels=labels, right=False)


def tenure_band(series: pd.Series) -> pd.Series:
    bins = [0, 6, 12, 24, 48, 120, 500]
    labels = ["0-5m", "6-11m", "12-23m", "24-47m", "48-119m", "120m+"]
    return pd.cut(series, bins=bins, labels=labels, right=False)


def delay_band(series: pd.Series) -> pd.Series:
    bins = [-1, 0, 7, 30, 120]
    labels = ["Sin retraso", "1-7 dias", "8-30 dias", "31+ dias"]
    return pd.cut(series, bins=bins, labels=labels)


def support_band(series: pd.Series) -> pd.Series:
    bins = [-1, 0, 1, 3, 100]
    labels = ["0", "1", "2-3", "4+"]
    return pd.cut(series, bins=bins, labels=labels)


def yes_no_flag(series: pd.Series) -> pd.Series:
    return series.fillna(0).astype(int).map({0: "No", 1: "Si"})


def chart_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", color=COLOR_GRID, alpha=0.6, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bdc7c2")
    ax.spines["bottom"].set_color("#bdc7c2")


@lru_cache(maxsize=1)
def load_churn() -> pd.DataFrame:
    return normalize_dates(pd.read_csv(prefer_processed_path("churn_target")))


@lru_cache(maxsize=1)
def load_clientes() -> pd.DataFrame:
    return normalize_dates(pd.read_csv(prefer_processed_path("clientes")))


@lru_cache(maxsize=1)
def load_facturacion() -> pd.DataFrame:
    return normalize_dates(pd.read_csv(prefer_processed_path("facturacion_mensual")))


@lru_cache(maxsize=1)
def load_soporte() -> pd.DataFrame:
    return normalize_dates(pd.read_csv(prefer_processed_path("interacciones_soporte")))


@lru_cache(maxsize=1)
def load_red() -> pd.DataFrame:
    return normalize_dates(pd.read_csv(prefer_processed_path("calidad_senal_zona_mensual")))


@lru_cache(maxsize=1)
def load_encuestas() -> pd.DataFrame:
    return normalize_dates(pd.read_csv(prefer_processed_path("encuestas_texto")))


@lru_cache(maxsize=1)
def master_data() -> pd.DataFrame:
    churn = load_churn().copy()
    clientes = load_clientes().copy()
    fact = load_facturacion().copy()

    master = churn.merge(
        clientes[
            [
                "cliente_id",
                "zona_id",
                "region",
                "tipo_zona",
                "edad",
                "sexo",
                "estado_civil",
                "num_lineas",
                "tipo_plan",
                "tipo_dispositivo",
                "ingreso_estimado",
                "antiguedad_meses",
                "descuento_activo",
            ]
        ].rename(
            columns={
                "zona_id": "zona_id_cliente",
                "num_lineas": "num_lineas_cliente",
                "tipo_plan": "tipo_plan_cliente",
            }
        ),
        on="cliente_id",
        how="left",
    )
    master = master.merge(
        fact[
            [
                "cliente_id",
                "fecha",
                "zona_id",
                "tipo_plan",
                "num_lineas",
                "cargo_base",
                "consumo_extra",
                "descuento_aplicado",
                "importe_total",
                "dias_retraso_pago",
                "impago_flag",
                "variacion_consumo_pct",
                "stress_calidad_lag",
                "incidencia_masiva_lag",
            ]
        ].rename(
            columns={
                "zona_id": "zona_id_fact",
                "tipo_plan": "tipo_plan_fact",
                "num_lineas": "num_lineas_fact",
            }
        ),
        on=["cliente_id", "fecha"],
        how="left",
    )

    master["zona_id"] = master["zona_id_fact"].fillna(master["zona_id_cliente"])
    master["tipo_plan"] = master["tipo_plan_fact"].fillna(master["tipo_plan_cliente"]).fillna("Desconocido")
    master["num_lineas"] = master["num_lineas_fact"].fillna(master["num_lineas_cliente"])
    master["edad_band"] = age_band(master["edad"])
    master["antiguedad_band"] = tenure_band(master["antiguedad_meses"])
    master["factura_band"] = safe_qcut(
        master["importe_total"],
        q=5,
        labels=["Muy baja", "Baja", "Media", "Alta", "Muy alta"],
    )
    master["ingreso_band"] = safe_qcut(
        master["ingreso_estimado"],
        q=5,
        labels=["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"],
    )
    master["retraso_band"] = delay_band(master["dias_retraso_pago"])
    master["impago_flag_label"] = yes_no_flag(master["impago_flag"])
    master["descuento_activo_label"] = yes_no_flag(master["descuento_activo"])
    master["churn_label"] = master["churn"].map({0: "No churn", 1: "Churn"})
    master = master.drop(columns=["zona_id_cliente", "zona_id_fact", "tipo_plan_cliente", "tipo_plan_fact", "num_lineas_cliente", "num_lineas_fact"])

    return master


@lru_cache(maxsize=1)
def support_data() -> pd.DataFrame:
    master = master_data()
    soporte = load_soporte().copy()
    soporte["motivo_norm"] = soporte["motivo"].fillna("Desconocido")
    soporte["canal_norm"] = soporte["canal"].fillna("Desconocido")
    soporte["fecha"] = soporte["mes"]
    soporte_agg = (
        soporte.groupby(["cliente_id", "fecha"], dropna=False)
        .agg(
            interacciones_totales=("interaccion_id", "size"),
            duracion_media=("duracion_min", "mean"),
            tasa_resuelto=("resuelto", "mean"),
            satisfaccion_media=("satisfaccion_post", "mean"),
            canal_top=("canal_norm", lambda s: s.mode().iat[0] if not s.mode().empty else "Desconocido"),
            motivo_top=("motivo_norm", lambda s: s.mode().iat[0] if not s.mode().empty else "Desconocido"),
            interacciones_baja=("motivo_norm", lambda s: int(s.str.contains("Baja", case=False, na=False).sum())),
            interacciones_facturacion=("motivo_norm", lambda s: int(s.str.contains("Fact", case=False, na=False).sum())),
        )
        .reset_index()
    )
    soporte_df = master.merge(soporte_agg, on=["cliente_id", "fecha"], how="left")
    soporte_df["interacciones_totales"] = soporte_df["interacciones_totales"].fillna(0)
    soporte_df["soporte_band"] = support_band(soporte_df["interacciones_totales"])
    soporte_df["tuvo_soporte"] = yes_no_flag((soporte_df["interacciones_totales"] > 0).astype(int))
    soporte_df["tuvo_baja_portabilidad"] = yes_no_flag((soporte_df["interacciones_baja"].fillna(0) > 0).astype(int))
    return soporte_df


@lru_cache(maxsize=1)
def network_data() -> pd.DataFrame:
    master = master_data()
    red = load_red().copy()
    red_df = master[["cliente_id", "fecha", "churn", "region", "tipo_zona", "tipo_plan", "zona_id"]].merge(
        red[
            [
                "zona_id",
                "fecha",
                "cobertura_4g_pct",
                "cobertura_5g_pct",
                "latencia_ms",
                "velocidad_media_mbps",
                "tasa_cortes_pct",
                "indice_calidad_global",
                "incidencia_masiva",
            ]
        ],
        on=["zona_id", "fecha"],
        how="left",
    )
    red_df["calidad_band"] = safe_qcut(
        red_df["indice_calidad_global"],
        q=5,
        labels=["Muy baja", "Baja", "Media", "Alta", "Muy alta"],
    )
    red_df["incidencia_masiva_label"] = yes_no_flag(red_df["incidencia_masiva"])
    return red_df


@lru_cache(maxsize=1)
def survey_data() -> pd.DataFrame:
    master = master_data()
    encuestas = load_encuestas().copy()
    encuestas_agg = (
        encuestas.groupby(["zona_id", "fecha"], dropna=False)
        .agg(
            encuestas_totales=("encuesta_id", "size"),
            nps_medio=("nps_0a10", "mean"),
            puntuacion_media=("puntuacion_general_1a5", "mean"),
            stress_calidad_medio=("stress_calidad", "mean"),
            sentimiento_medio=("sent_text_latente", "mean"),
        )
        .reset_index()
    )
    survey_df = master[["cliente_id", "fecha", "churn", "region", "tipo_zona", "tipo_plan", "zona_id"]].merge(
        encuestas_agg,
        on=["zona_id", "fecha"],
        how="left",
    )
    survey_df["nps_band"] = pd.cut(
        survey_df["nps_medio"],
        bins=[-1, 3, 6, 8, 10],
        labels=["Muy bajo", "Bajo", "Medio", "Alto"],
    )
    survey_df["stress_band"] = safe_qcut(
        survey_df["stress_calidad_medio"],
        q=4,
        labels=["Bajo", "Medio-bajo", "Medio-alto", "Alto"],
    )
    survey_df["sentimiento_band"] = pd.cut(
        survey_df["sentimiento_medio"],
        bins=[-10, -0.5, 0.0, 0.5, 10],
        labels=["Muy negativo", "Negativo", "Neutro/positivo", "Muy positivo"],
    )
    return survey_df


def hypotheses_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Clientes", "Los clientes con menor antiguedad tienen mas churn", "Comparar churn por banda de antiguedad y edad."],
            ["Clientes", "El mix de plan y dispositivo condiciona el riesgo", "Ver churn por tipo_plan y tipo_dispositivo."],
            ["Facturacion", "El impago y el retraso en pago preceden el churn", "Medir churn por impago_flag y retraso_band."],
            ["Facturacion", "Cambios bruscos de consumo anticipan fuga", "Comparar churn por variacion_consumo_pct y factura_band."],
            ["Facturacion", "Los descuentos pueden retener o atraer clientes fragiles", "Cruzar churn con descuento_activo y descuento_aplicado."],
            ["Soporte", "Un mayor volumen de interacciones senala friccion", "Analizar churn por soporte_band y tuvo_soporte."],
            ["Soporte", "Los contactos por baja o portabilidad son una alarma temprana", "Comparar churn con tuvo_baja_portabilidad y motivo_top."],
            ["Soporte", "Una baja resolucion empeora la retencion", "Medir churn frente a tasa_resuelto y satisfaccion_media."],
            ["Red", "La peor calidad de red se asocia a mayor churn", "Evaluar churn por calidad_band e indice_calidad_global."],
            ["Red", "Las incidencias masivas elevan el riesgo", "Contrastar churn por incidencia_masiva_label y region."],
            ["Encuestas", "Un NPS bajo es un predictor de fuga", "Comparar churn por nps_band y nps_medio."],
            ["Encuestas", "Sentimiento negativo y stress alto capturan malestar real", "Revisar churn por stress_band y sentimiento_band."],
        ],
        columns=["dataset", "hipotesis", "como_responderla"],
    )


def apply_global_filters(
    df: pd.DataFrame,
    churn_value: str,
    region: str,
    tipo_zona: str,
    tipo_plan: str,
    month_start: str,
    month_end: str,
) -> pd.DataFrame:
    filtered = df.copy()
    if churn_value != ALL_VALUE and "churn" in filtered.columns:
        filtered = filtered.loc[filtered["churn"] == int(churn_value)]
    if region != ALL_VALUE and "region" in filtered.columns:
        filtered = filtered.loc[filtered["region"] == region]
    if tipo_zona != ALL_VALUE and "tipo_zona" in filtered.columns:
        filtered = filtered.loc[filtered["tipo_zona"] == tipo_zona]
    if tipo_plan != ALL_VALUE and "tipo_plan" in filtered.columns:
        filtered = filtered.loc[filtered["tipo_plan"] == tipo_plan]
    if "fecha" in filtered.columns:
        months = filtered["fecha"].dt.strftime("%Y-%m")
        if month_start != MONTH_ALL:
            filtered = filtered.loc[months >= month_start]
            months = filtered["fecha"].dt.strftime("%Y-%m")
        if month_end != MONTH_ALL:
            filtered = filtered.loc[months <= month_end]
    return filtered.reset_index(drop=True)


def metric_text(df: pd.DataFrame) -> tuple[str, str, str]:
    rows = f"{len(df):,}"
    clients = f"{df['cliente_id'].nunique():,}" if "cliente_id" in df.columns else "0"
    churn_rate = "Sin datos"
    if len(df) and "churn" in df.columns:
        churn_rate = f"{df['churn'].mean() * 100:.2f}%"
    return rows, clients, churn_rate


def collapse_to_clients(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ordered = df.sort_values(["cliente_id", "fecha"] if "fecha" in df.columns else ["cliente_id"]).copy()

    def last_valid(series: pd.Series):
        non_null = series.dropna()
        return non_null.iloc[-1] if not non_null.empty else pd.NA

    agg_map = {"churn": "max"}
    keep_last = [
        "region",
        "tipo_zona",
        "tipo_plan",
        "edad",
        "edad_band",
        "sexo",
        "estado_civil",
        "tipo_dispositivo",
        "ingreso_estimado",
        "ingreso_band",
        "antiguedad_meses",
        "antiguedad_band",
        "descuento_activo",
        "descuento_activo_label",
        "retraso_band",
        "impago_flag_label",
        "factura_band",
    ]
    for col in keep_last:
        if col in ordered.columns:
            agg_map[col] = last_valid

    collapsed = ordered.groupby("cliente_id", dropna=False).agg(agg_map).reset_index()
    return collapsed


def bar_churn_rate(fig_title: str, df: pd.DataFrame, category: str, top_n: int = 10) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    if df.empty or category not in df.columns:
        ax.text(0.5, 0.5, "No hay datos para esta vista.", ha="center", va="center")
        ax.axis("off")
        return fig

    plot_df = df[[category, "churn"]].dropna()
    plot_df[category] = plot_df[category].astype(str)
    grouped = plot_df.groupby(category, dropna=False).agg(churn_rate=("churn", "mean"), clientes=("churn", "size"))
    if pd.api.types.is_categorical_dtype(df[category]):
        ordered_index = [str(x) for x in df[category].cat.categories if str(x) in grouped.index.astype(str).tolist()]
        grouped.index = grouped.index.astype(str)
        grouped = grouped.reindex(ordered_index).dropna(subset=["churn_rate"])
    else:
        grouped.index = grouped.index.astype(str)
        grouped = grouped.sort_values("churn_rate", ascending=False).head(top_n).sort_values("churn_rate", ascending=True)
    ax.barh(grouped.index, grouped["churn_rate"] * 100, color=COLOR_CHURN, alpha=0.9)
    ax.set_xlabel("Tasa de churn (%)")
    ax.set_title(fig_title)
    chart_style(ax)
    fig.patch.set_facecolor(BG_FIG)
    fig.tight_layout()
    return fig


def line_churn_rate(fig_title: str, df: pd.DataFrame, value_col: str = "churn") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    if df.empty or "fecha" not in df.columns:
        ax.text(0.5, 0.5, "No hay datos para esta vista.", ha="center", va="center")
        ax.axis("off")
        return fig

    plot_df = (
        df.groupby("fecha", dropna=False)[value_col]
        .mean()
        .reset_index()
        .sort_values("fecha")
    )
    ax.plot(plot_df["fecha"], plot_df[value_col] * 100, color=COLOR_CHURN, linewidth=2.6, marker="o")
    ax.set_ylabel("Tasa de churn (%)")
    ax.set_title(fig_title)
    chart_style(ax)
    fig.patch.set_facecolor(BG_FIG)
    fig.tight_layout()
    return fig


def compare_distribution(fig_title: str, df: pd.DataFrame, numeric_col: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    if df.empty or numeric_col not in df.columns or not is_numeric_dtype(df[numeric_col]):
        ax.text(0.5, 0.5, "No hay datos numericos para esta vista.", ha="center", va="center")
        ax.axis("off")
        return fig

    safe = df.loc[df["churn"] == 0, numeric_col].dropna()
    churn = df.loc[df["churn"] == 1, numeric_col].dropna()
    if safe.empty and churn.empty:
        ax.text(0.5, 0.5, "No hay datos numericos para esta vista.", ha="center", va="center")
        ax.axis("off")
        return fig

    ax.hist(safe, bins=30, alpha=0.55, color=COLOR_SAFE, label="No churn")
    ax.hist(churn, bins=30, alpha=0.65, color=COLOR_CHURN, label="Churn")
    ax.set_title(fig_title)
    ax.set_ylabel("Frecuencia")
    ax.legend(frameon=False)
    chart_style(ax)
    fig.patch.set_facecolor(BG_FIG)
    fig.tight_layout()
    return fig


def box_by_churn(fig_title: str, df: pd.DataFrame, numeric_col: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    if df.empty or numeric_col not in df.columns:
        ax.text(0.5, 0.5, "No hay datos para esta vista.", ha="center", va="center")
        ax.axis("off")
        return fig

    safe = df.loc[df["churn"] == 0, numeric_col].dropna()
    churn = df.loc[df["churn"] == 1, numeric_col].dropna()
    bp = ax.boxplot([safe, churn], labels=["No churn", "Churn"], patch_artist=True)
    for patch, color in zip(bp["boxes"], [COLOR_SAFE, COLOR_CHURN]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title(fig_title)
    chart_style(ax)
    fig.patch.set_facecolor(BG_FIG)
    fig.tight_layout()
    return fig


def grouped_summary_table(df: pd.DataFrame, group_col: str, top_n: int = 12) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "clientes", "churn_rate_pct"])
    table = (
        df[[group_col, "churn"]]
        .dropna()
        .assign(**{group_col: lambda x: x[group_col].astype(str)})
        .groupby(group_col, dropna=False)
        .agg(clientes=("churn", "size"), churn_rate_pct=("churn", lambda s: round(s.mean() * 100, 2)))
        .sort_values(["clientes", "churn_rate_pct"], ascending=[False, False])
        .head(top_n)
        .reset_index()
    )
    return table


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def two_prop_test(success_a: int, n_a: int, success_b: int, n_b: int) -> tuple[float, float]:
    if min(n_a, n_b) == 0:
        return np.nan, np.nan
    p_pool = (success_a + success_b) / (n_a + n_b)
    se = sqrt(max(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b), 1e-12))
    z_value = ((success_a / n_a) - (success_b / n_b)) / se
    p_value = 2 * (1 - normal_cdf(abs(z_value)))
    return z_value, p_value


def mean_diff_test(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    a = a.dropna()
    b = b.dropna()
    if min(len(a), len(b)) < 2:
        return np.nan, np.nan, np.nan
    mean_diff = a.mean() - b.mean()
    var_term = (a.var(ddof=1) / len(a)) + (b.var(ddof=1) / len(b))
    se = sqrt(max(var_term, 1e-12))
    z_value = mean_diff / se
    pooled_std = sqrt(max(((a.var(ddof=1) + b.var(ddof=1)) / 2), 1e-12))
    effect = mean_diff / pooled_std
    p_value = 2 * (1 - normal_cdf(abs(z_value)))
    return mean_diff, effect, p_value


def top_category_insight(df: pd.DataFrame, category: str, label: str, entity_label: str = "observaciones") -> str:
    if df.empty or category not in df.columns:
        return f"No hay datos suficientes para evaluar {label.lower()}."
    grouped = (
        df[[category, "churn"]]
        .dropna()
        .assign(**{category: lambda x: x[category].astype(str)})
        .groupby(category, dropna=False)
        .agg(clientes=("churn", "size"), churns=("churn", "sum"), churn_rate=("churn", "mean"))
        .query("clientes >= 30")
    )
    if grouped.empty:
        return f"No hay volumen suficiente para extraer una lectura estable sobre {label.lower()}."
    baseline_rate = df["churn"].mean()
    top = grouped.sort_values("churn_rate", ascending=False).iloc[0]
    low = grouped.sort_values("churn_rate", ascending=True).iloc[0]
    z_value, p_value = two_prop_test(int(top["churns"]), int(top["clientes"]), int(low["churns"]), int(low["clientes"]))
    lift = (top["churn_rate"] / baseline_rate - 1) * 100 if baseline_rate > 0 else np.nan
    return (
        f"{label}: el segmento con mas riesgo es `{grouped.sort_values('churn_rate', ascending=False).index[0]}` "
        f"con churn {top['churn_rate'] * 100:.2f}% sobre {int(top['clientes'])} {entity_label}. "
        f"Eso implica un lift de {lift:.1f}% frente a la media filtrada. "
        f"Comparado con el segmento mas estable (`{grouped.sort_values('churn_rate').index[0]}`), "
        f"la diferencia tiene z={z_value:.2f} y p={p_value:.4f}."
    )


def numeric_insight(df: pd.DataFrame, numeric_col: str, label: str) -> str:
    if df.empty or numeric_col not in df.columns:
        return f"No hay datos suficientes para evaluar {label.lower()}."
    churn = df.loc[df["churn"] == 1, numeric_col]
    safe = df.loc[df["churn"] == 0, numeric_col]
    mean_diff, effect, p_value = mean_diff_test(churn, safe)
    if pd.isna(mean_diff):
        return f"No hay datos suficientes para evaluar {label.lower()}."
    direction = "mayor" if mean_diff > 0 else "menor"
    return (
        f"{label}: los churners muestran un valor medio {direction} en {abs(mean_diff):.2f} unidades "
        f"respecto a no churn. El tamano de efecto estandarizado es {effect:.2f} y la evidencia aproximada "
        f"da p={p_value:.4f}."
    )


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.h2("Churn Lab"),
            ui.p("Una app guiada por hipotesis, no un generador de plots arbitrarios."),
            class_="sidebar-hero",
        ),
        ui.input_select("churn_filter", "Segmento", choices={ALL_VALUE: "Todos", "1": "Solo churn", "0": "Solo no churn"}, selected=ALL_VALUE),
        ui.input_select("region_filter", "Region", choices={ALL_VALUE: "Todas"}, selected=ALL_VALUE),
        ui.input_select("zona_filter", "Tipo de zona", choices={ALL_VALUE: "Todas"}, selected=ALL_VALUE),
        ui.input_select("plan_filter", "Tipo de plan", choices={ALL_VALUE: "Todos"}, selected=ALL_VALUE),
        ui.input_select("month_start", "Mes inicial", choices={MONTH_ALL: "Todos"}, selected=MONTH_ALL),
        ui.input_select("month_end", "Mes final", choices={MONTH_ALL: "Todos"}, selected=MONTH_ALL),
        ui.hr(),
        ui.markdown(
            f"""
**Leyenda de color**

- `No churn`: gris azulado
- `Churn`: terracota
- `Analisis agregado`: verde petroleo
"""
        ),
        width=340,
        open="desktop",
    ),
    ui.tags.head(
        ui.tags.style(
            """
            :root {
                --bg: #f2efe8;
                --panel: rgba(255, 255, 255, 0.86);
                --panel-strong: rgba(255, 255, 255, 0.94);
                --ink: #14212b;
                --muted: #5d6b74;
                --line: rgba(20, 33, 43, 0.10);
                --accent: #0f766e;
                --danger: #c65d4b;
                --safe: #8091a0;
                --shadow: 0 18px 40px rgba(20, 33, 43, 0.07);
            }
            body {
                background:
                    radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 26%),
                    radial-gradient(circle at right 18%, rgba(198, 93, 75, 0.08), transparent 24%),
                    linear-gradient(180deg, #f8f6f1 0%, var(--bg) 100%);
                color: var(--ink);
                font-family: "IBM Plex Sans", "Trebuchet MS", sans-serif;
            }
            h1, h2, h3, h4, .sidebar-hero {
                font-family: "Aptos Display", "Palatino Linotype", serif;
                letter-spacing: 0.01em;
            }
            .sidebar-hero {
                margin-bottom: 0.8rem;
                padding: 1rem 1rem 0.2rem 1rem;
                border-radius: 20px;
                background: linear-gradient(145deg, rgba(20, 33, 43, 0.96), rgba(15, 118, 110, 0.88));
                color: #f7faf9;
                box-shadow: var(--shadow);
            }
            .sidebar-hero p {
                color: rgba(247, 250, 249, 0.82);
                line-height: 1.4;
            }
            .bslib-sidebar-layout > .main {
                padding: 1.25rem 1.4rem 2rem 1.4rem;
            }
            .card {
                border: 1px solid var(--line);
                border-radius: 20px;
                background: var(--panel);
                backdrop-filter: blur(10px);
                box-shadow: var(--shadow);
            }
            .card-header {
                background: transparent;
                border-bottom: 1px solid var(--line);
                font-weight: 600;
                color: var(--ink);
            }
            .shiny-text-output {
                white-space: pre-wrap;
                line-height: 1.5;
            }
            .metric-box {
                padding: 1rem 1.1rem;
                border-radius: 18px;
                border: 1px solid var(--line);
                background: var(--panel-strong);
                box-shadow: var(--shadow);
                min-height: 114px;
            }
            .metric-label {
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.09em;
                font-size: 0.75rem;
                margin-bottom: 0.45rem;
            }
            .metric-value {
                font-size: 1.95rem;
                font-weight: 700;
                line-height: 1.05;
            }
            .metric-note {
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.35;
                margin-top: 0.35rem;
            }
            """
        )
    ),
    ui.h1("Analisis de churn guiado por hipotesis"),
    ui.p("Cada pestaña responde preguntas de negocio concretas para un dataset distinto, con codificacion visual consistente."),
    ui.layout_columns(
        ui.div(ui.div("Observaciones", class_="metric-label"), ui.output_text("metric_rows"), ui.div("Unidad base: cliente-mes en vistas dinamicas.", class_="metric-note"), class_="metric-box"),
        ui.div(ui.div("Clientes unicos", class_="metric-label"), ui.output_text("metric_clients"), ui.div("Clientes distintos en la muestra filtrada.", class_="metric-note"), class_="metric-box"),
        ui.div(ui.div("Tasa de churn", class_="metric-label"), ui.output_text("metric_churn"), ui.div("Porcentaje de clientes unicos que churnean alguna vez en la muestra filtrada.", class_="metric-note"), class_="metric-box"),
        col_widths=(4, 4, 4),
    ),
    ui.navset_tab(
        ui.nav_panel(
            "Resumen",
            ui.layout_columns(
                ui.card(ui.card_header("Evolucion temporal del churn"), ui.output_plot("overview_trend", height="360px")),
                ui.card(ui.card_header("Donde se concentra mas churn"), ui.output_plot("overview_region", height="360px")),
                col_widths=(6, 6),
            ),
            ui.card(ui.card_header("Lectura ejecutiva"), ui.output_text_verbatim("overview_insight")),
            ui.card(ui.card_header("Churn por tipo de plan"), ui.output_data_frame("overview_table")),
        ),
        ui.nav_panel(
            "Clientes",
            ui.layout_columns(
                ui.card(ui.card_header("Hipotesis: menor antiguedad, mayor churn"), ui.output_plot("clients_tenure_plot", height="360px")),
                ui.card(ui.card_header("Hipotesis: ciertos perfiles etarios son mas fragiles"), ui.output_plot("clients_age_plot", height="360px")),
                col_widths=(6, 6),
            ),
            ui.card(ui.card_header("Lectura ejecutiva"), ui.output_text_verbatim("clients_insight")),
            ui.card(ui.card_header("Segmentos estructurales mas expuestos"), ui.output_data_frame("clients_table")),
        ),
        ui.nav_panel(
            "Facturacion",
            ui.layout_columns(
                ui.card(ui.card_header("Hipotesis: impago y retraso elevan el churn"), ui.output_plot("billing_delay_plot", height="360px")),
                ui.card(ui.card_header("Hipotesis: la factura total diferencia churners"), ui.output_plot("billing_amount_plot", height="360px")),
                col_widths=(6, 6),
            ),
            ui.card(ui.card_header("Lectura ejecutiva"), ui.output_text_verbatim("billing_insight")),
            ui.card(ui.card_header("Resumen de riesgo financiero"), ui.output_data_frame("billing_table")),
        ),
        ui.nav_panel(
            "Soporte",
            ui.layout_columns(
                ui.card(ui.card_header("Hipotesis: mas contactos, mas friccion"), ui.output_plot("support_volume_plot", height="360px")),
                ui.card(ui.card_header("Hipotesis: baja o portabilidad anticipa churn"), ui.output_plot("support_motive_plot", height="360px")),
                col_widths=(6, 6),
            ),
            ui.card(ui.card_header("Lectura ejecutiva"), ui.output_text_verbatim("support_insight")),
            ui.card(ui.card_header("Canales y motivos mas sensibles"), ui.output_data_frame("support_table")),
        ),
        ui.nav_panel(
            "Red",
            ui.layout_columns(
                ui.card(ui.card_header("Hipotesis: peor calidad, mayor churn"), ui.output_plot("network_quality_plot", height="360px")),
                ui.card(ui.card_header("Hipotesis: incidencias masivas aumentan el riesgo"), ui.output_plot("network_incidence_plot", height="360px")),
                col_widths=(6, 6),
            ),
            ui.card(ui.card_header("Lectura ejecutiva"), ui.output_text_verbatim("network_insight")),
            ui.card(ui.card_header("Resumen por calidad y region"), ui.output_data_frame("network_table")),
        ),
        ui.nav_panel(
            "Encuestas",
            ui.layout_columns(
                ui.card(ui.card_header("Hipotesis: NPS bajo se asocia a fuga"), ui.output_plot("survey_nps_plot", height="360px")),
                ui.card(ui.card_header("Hipotesis: el sentimiento y stress capturan malestar"), ui.output_plot("survey_sentiment_plot", height="360px")),
                col_widths=(6, 6),
            ),
            ui.card(ui.card_header("Lectura ejecutiva"), ui.output_text_verbatim("survey_insight")),
            ui.card(ui.card_header("Lectura sintetica de percepcion"), ui.output_data_frame("survey_table")),
        ),
        ui.nav_panel(
            "Hipotesis",
            ui.card(
                ui.card_header("Preguntas que puedes responder con esta app"),
                ui.output_data_frame("hypotheses_table"),
            ),
        ),
    ),
    title="Churn Lab",
    fillable=True,
)


def server(input, output, session):
    @reactive.calc
    def master_df() -> pd.DataFrame:
        return master_data()

    @reactive.calc
    def filtered_master() -> pd.DataFrame:
        return apply_global_filters(
            master_df(),
            input.churn_filter(),
            input.region_filter(),
            input.zona_filter(),
            input.plan_filter(),
            input.month_start(),
            input.month_end(),
        )

    @reactive.calc
    def filtered_clients() -> pd.DataFrame:
        return collapse_to_clients(filtered_master())

    @reactive.calc
    def filtered_support() -> pd.DataFrame:
        return apply_global_filters(
            support_data(),
            input.churn_filter(),
            input.region_filter(),
            input.zona_filter(),
            input.plan_filter(),
            input.month_start(),
            input.month_end(),
        )

    @reactive.calc
    def filtered_network() -> pd.DataFrame:
        return apply_global_filters(
            network_data(),
            input.churn_filter(),
            input.region_filter(),
            input.zona_filter(),
            input.plan_filter(),
            input.month_start(),
            input.month_end(),
        )

    @reactive.calc
    def filtered_survey() -> pd.DataFrame:
        return apply_global_filters(
            survey_data(),
            input.churn_filter(),
            input.region_filter(),
            input.zona_filter(),
            input.plan_filter(),
            input.month_start(),
            input.month_end(),
        )

    @reactive.effect
    def _sync_filters() -> None:
        df = master_df()
        ui.update_select(
            "region_filter",
            choices={ALL_VALUE: "Todas", **{x: x for x in sorted(df["region"].dropna().astype(str).unique())}},
            selected=ALL_VALUE,
        )
        ui.update_select(
            "zona_filter",
            choices={ALL_VALUE: "Todas", **{x: x for x in sorted(df["tipo_zona"].dropna().astype(str).unique())}},
            selected=ALL_VALUE,
        )
        ui.update_select(
            "plan_filter",
            choices={ALL_VALUE: "Todos", **{x: x for x in sorted(df["tipo_plan"].dropna().astype(str).unique())}},
            selected=ALL_VALUE,
        )
        month_map = month_choices(df)
        ui.update_select("month_start", choices=month_map, selected=MONTH_ALL)
        ui.update_select("month_end", choices=month_map, selected=MONTH_ALL)

    @output
    @render.text
    def metric_rows() -> str:
        return metric_text(filtered_master())[0]

    @output
    @render.text
    def metric_clients() -> str:
        return metric_text(filtered_master())[1]

    @output
    @render.text
    def metric_churn() -> str:
        return metric_text(filtered_clients())[2]

    @output
    @render.plot(alt="Tendencia temporal del churn")
    def overview_trend():
        return line_churn_rate("Tasa de churn a lo largo del tiempo", filtered_master())

    @output
    @render.plot(alt="Churn por region")
    def overview_region():
        return bar_churn_rate("Tasa de churn por region (clientes unicos)", filtered_clients(), "region", top_n=8)

    @output
    @render.data_frame
    def overview_table():
        return render.DataGrid(grouped_summary_table(filtered_clients(), "tipo_plan"), width="100%", height="340px", filters=True)

    @output
    @render.text
    def overview_insight() -> str:
        df = filtered_clients()
        if df.empty:
            return "No hay datos con los filtros actuales."
        region_text = top_category_insight(df, "region", "Region", entity_label="clientes")
        plan_text = top_category_insight(df, "tipo_plan", "Plan", entity_label="clientes")
        return "\n\n".join([region_text, plan_text])

    @output
    @render.plot(alt="Churn por antiguedad")
    def clients_tenure_plot():
        return bar_churn_rate("Churn por banda de antiguedad (clientes unicos)", filtered_clients(), "antiguedad_band", top_n=10)

    @output
    @render.plot(alt="Churn por edad")
    def clients_age_plot():
        return bar_churn_rate("Churn por banda de edad (clientes unicos)", filtered_clients(), "edad_band", top_n=10)

    @output
    @render.data_frame
    def clients_table():
        table = (
            filtered_clients()[["tipo_plan", "tipo_dispositivo", "churn"]]
            .dropna()
            .groupby(["tipo_plan", "tipo_dispositivo"], dropna=False)
            .agg(clientes=("churn", "size"), churn_rate_pct=("churn", lambda s: round(s.mean() * 100, 2)))
            .sort_values(["churn_rate_pct", "clientes"], ascending=[False, False])
            .head(15)
            .reset_index()
        )
        return render.DataGrid(table, width="100%", height="360px", filters=True)

    @output
    @render.text
    def clients_insight() -> str:
        df = filtered_clients()
        return "\n\n".join(
            [
                top_category_insight(df, "antiguedad_band", "Antiguedad", entity_label="clientes"),
                top_category_insight(df, "edad_band", "Edad", entity_label="clientes"),
            ]
        )

    @output
    @render.plot(alt="Churn por retraso de pago")
    def billing_delay_plot():
        return bar_churn_rate("Churn por retraso de pago", filtered_master(), "retraso_band", top_n=10)

    @output
    @render.plot(alt="Distribucion de importe total por churn")
    def billing_amount_plot():
        return compare_distribution("Distribucion del importe total por churn", filtered_master(), "importe_total")

    @output
    @render.data_frame
    def billing_table():
        table = (
            filtered_master()[["cliente_id", "impago_flag_label", "retraso_band", "churn"]]
            .dropna()
            .groupby(["impago_flag_label", "retraso_band"], dropna=False)
            .agg(
                observaciones_cliente_mes=("churn", "size"),
                clientes_unicos=("cliente_id", "nunique"),
                churn_rate_pct=("churn", lambda s: round(s.mean() * 100, 2)),
            )
            .sort_values(["churn_rate_pct", "observaciones_cliente_mes"], ascending=[False, False])
            .reset_index()
        )
        return render.DataGrid(table, width="100%", height="360px", filters=True)

    @output
    @render.text
    def billing_insight() -> str:
        df = filtered_master()
        return "\n\n".join(
            [
                top_category_insight(df, "retraso_band", "Retraso de pago", entity_label="observaciones cliente-mes"),
                numeric_insight(df, "importe_total", "Importe total"),
            ]
        )

    @output
    @render.plot(alt="Churn por intensidad de soporte")
    def support_volume_plot():
        return bar_churn_rate("Churn por volumen de interacciones", filtered_support(), "soporte_band", top_n=10)

    @output
    @render.plot(alt="Churn por alerta de baja o portabilidad")
    def support_motive_plot():
        return bar_churn_rate("Churn cuando aparece baja o portabilidad", filtered_support(), "tuvo_baja_portabilidad", top_n=5)

    @output
    @render.data_frame
    def support_table():
        table = (
            filtered_support()[["cliente_id", "canal_top", "motivo_top", "churn"]]
            .dropna()
            .groupby(["canal_top", "motivo_top"], dropna=False)
            .agg(
                observaciones_cliente_mes=("churn", "size"),
                clientes_unicos=("cliente_id", "nunique"),
                churn_rate_pct=("churn", lambda s: round(s.mean() * 100, 2)),
            )
            .sort_values(["observaciones_cliente_mes", "churn_rate_pct"], ascending=[False, False])
            .head(20)
            .reset_index()
        )
        return render.DataGrid(table, width="100%", height="360px", filters=True)

    @output
    @render.text
    def support_insight() -> str:
        df = filtered_support()
        return "\n\n".join(
            [
                top_category_insight(df, "soporte_band", "Volumen de soporte", entity_label="observaciones cliente-mes"),
                top_category_insight(df, "tuvo_baja_portabilidad", "Alerta de baja o portabilidad", entity_label="observaciones cliente-mes"),
            ]
        )

    @output
    @render.plot(alt="Churn por calidad de red")
    def network_quality_plot():
        return bar_churn_rate("Churn por banda de calidad de red", filtered_network(), "calidad_band", top_n=10)

    @output
    @render.plot(alt="Churn por incidencias masivas")
    def network_incidence_plot():
        return bar_churn_rate("Churn con y sin incidencia masiva", filtered_network(), "incidencia_masiva_label", top_n=5)

    @output
    @render.data_frame
    def network_table():
        table = (
            filtered_network()[["cliente_id", "region", "calidad_band", "churn"]]
            .dropna()
            .groupby(["region", "calidad_band"], dropna=False)
            .agg(
                observaciones_cliente_mes=("churn", "size"),
                clientes_unicos=("cliente_id", "nunique"),
                churn_rate_pct=("churn", lambda s: round(s.mean() * 100, 2)),
            )
            .sort_values(["churn_rate_pct", "observaciones_cliente_mes"], ascending=[False, False])
            .reset_index()
        )
        return render.DataGrid(table, width="100%", height="360px", filters=True)

    @output
    @render.text
    def network_insight() -> str:
        df = filtered_network()
        return "\n\n".join(
            [
                top_category_insight(df, "calidad_band", "Calidad de red", entity_label="observaciones cliente-mes"),
                top_category_insight(df, "incidencia_masiva_label", "Incidencia masiva", entity_label="observaciones cliente-mes"),
            ]
        )

    @output
    @render.plot(alt="Churn por banda de NPS")
    def survey_nps_plot():
        return bar_churn_rate("Churn por banda de NPS", filtered_survey(), "nps_band", top_n=10)

    @output
    @render.plot(alt="Churn por sentimiento")
    def survey_sentiment_plot():
        return bar_churn_rate("Churn por sentimiento agregado", filtered_survey(), "sentimiento_band", top_n=10)

    @output
    @render.data_frame
    def survey_table():
        table = (
            filtered_survey()[["cliente_id", "nps_band", "stress_band", "churn"]]
            .dropna()
            .groupby(["nps_band", "stress_band"], dropna=False)
            .agg(
                observaciones_cliente_mes=("churn", "size"),
                clientes_unicos=("cliente_id", "nunique"),
                churn_rate_pct=("churn", lambda s: round(s.mean() * 100, 2)),
            )
            .sort_values(["churn_rate_pct", "observaciones_cliente_mes"], ascending=[False, False])
            .reset_index()
        )
        return render.DataGrid(table, width="100%", height="360px", filters=True)

    @output
    @render.text
    def survey_insight() -> str:
        df = filtered_survey()
        return "\n\n".join(
            [
                top_category_insight(df, "nps_band", "NPS", entity_label="observaciones cliente-mes"),
                top_category_insight(df, "sentimiento_band", "Sentimiento", entity_label="observaciones cliente-mes"),
            ]
        )

    @output
    @render.data_frame
    def hypotheses_table():
        return render.DataGrid(hypotheses_data(), width="100%", height="420px", filters=True)


app = App(app_ui, server)
