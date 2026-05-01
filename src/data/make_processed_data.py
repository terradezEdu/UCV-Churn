from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


@dataclass
class QualityRecord:
    dataset: str
    rows_raw: int
    rows_clean: int
    dropped_rows: int
    exact_duplicates_raw: int
    key_duplicates_raw: int
    key_duplicates_clean: int
    notes: str


def month_start(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp()


def tidy_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df


def impute_group_median(df: pd.DataFrame, column: str, group_col: str) -> pd.Series:
    series = df[column].copy()
    group_median = df.groupby(group_col, dropna=False)[column].transform("median")
    series = series.fillna(group_median)
    return series.fillna(series.median())


def drop_best_record_per_key(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    scored = df.copy()
    scored["_row_completeness"] = scored.notna().sum(axis=1)
    scored = scored.sort_values(key_cols + ["_row_completeness"], ascending=[True] * len(key_cols) + [False])
    scored = scored.drop_duplicates(subset=key_cols, keep="first")
    return scored.drop(columns="_row_completeness")


def summarize(dataset: str, raw_df: pd.DataFrame, clean_df: pd.DataFrame, key_cols: list[str], notes: str) -> QualityRecord:
    return QualityRecord(
        dataset=dataset,
        rows_raw=len(raw_df),
        rows_clean=len(clean_df),
        dropped_rows=len(raw_df) - len(clean_df),
        exact_duplicates_raw=int(raw_df.duplicated().sum()),
        key_duplicates_raw=int(raw_df.duplicated(key_cols).sum()),
        key_duplicates_clean=int(clean_df.duplicated(key_cols).sum()),
        notes=notes,
    )


def clean_churn_target() -> tuple[pd.DataFrame, QualityRecord]:
    raw = pd.read_csv(RAW_DIR / "churn_target.csv")
    df = tidy_strings(raw)
    df["fecha"] = month_start(df["fecha"])
    df["churn"] = pd.to_numeric(df["churn"], errors="coerce")
    df = df.loc[df["cliente_id"].notna() & df["fecha"].notna()]
    df = df.loc[df["churn"].isin([0, 1])].copy()
    df["churn"] = df["churn"].astype(int)
    df = drop_best_record_per_key(df.drop_duplicates(), ["cliente_id", "fecha"])
    df = df.sort_values(["cliente_id", "fecha"]).reset_index(drop=True)
    return df, summarize(
        "churn_target",
        raw,
        df,
        ["cliente_id", "fecha"],
        "El target ya venia bastante limpio; se normalizaron fechas y se garantizo unicidad cliente-mes.",
    )


def clean_clientes() -> tuple[pd.DataFrame, QualityRecord]:
    raw = pd.read_csv(RAW_DIR / "clientes.csv")
    df = tidy_strings(raw).drop_duplicates()

    numeric_cols = ["edad", "num_lineas", "ingreso_estimado", "antiguedad_meses", "poblacion_zona"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[~df["edad"].between(18, 100), "edad"] = np.nan
    df.loc[df["num_lineas"] < 1, "num_lineas"] = np.nan
    df.loc[df["ingreso_estimado"] <= 0, "ingreso_estimado"] = np.nan
    df.loc[~df["antiguedad_meses"].between(0, 240), "antiguedad_meses"] = np.nan

    for col in ["edad", "ingreso_estimado", "antiguedad_meses", "num_lineas"]:
        df[col] = impute_group_median(df, col, "tipo_plan")

    for col in ["sexo", "estado_civil", "tipo_dispositivo", "tipo_plan", "region", "tipo_zona"]:
        df[col] = df[col].fillna("Desconocido")

    df["descuento_activo"] = (
        pd.to_numeric(df["descuento_activo"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    )

    df = drop_best_record_per_key(df, ["cliente_id"])
    df = df.sort_values("cliente_id").reset_index(drop=True)
    return df, summarize(
        "clientes",
        raw,
        df,
        ["cliente_id"],
        "Se eliminaron duplicados, se corrigieron antiguedades imposibles y se imputaron faltantes estructurales por tipo_plan.",
    )


def clean_facturacion(clientes_clean: pd.DataFrame) -> tuple[pd.DataFrame, QualityRecord]:
    raw = pd.read_csv(RAW_DIR / "facturacion_mensual.csv")
    df = tidy_strings(raw).drop_duplicates()

    df["fecha"] = month_start(df["fecha"])
    for col in [
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
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = drop_best_record_per_key(df, ["cliente_id", "fecha"])
    df = df.merge(
        clientes_clean[["cliente_id", "tipo_plan", "num_lineas", "zona_id"]].rename(
            columns={
                "tipo_plan": "tipo_plan_cliente",
                "num_lineas": "num_lineas_cliente",
                "zona_id": "zona_id_cliente",
            }
        ),
        on="cliente_id",
        how="left",
    )

    df["tipo_plan"] = df["tipo_plan"].fillna(df["tipo_plan_cliente"]).fillna("Desconocido")
    df["num_lineas"] = df["num_lineas"].fillna(df["num_lineas_cliente"])
    df["zona_id"] = df["zona_id"].fillna(df["zona_id_cliente"])
    df["num_lineas"] = df["num_lineas"].fillna(df["num_lineas"].median()).clip(lower=1)

    reconstructed_total = df["cargo_base"].fillna(0) + df["consumo_extra"].fillna(0) - df["descuento_aplicado"].fillna(0)
    df["importe_total"] = df["importe_total"].fillna(reconstructed_total).clip(lower=0)
    df["dias_retraso_pago"] = df["dias_retraso_pago"].fillna(0).clip(lower=0)
    df["impago_flag"] = df["impago_flag"].fillna((df["dias_retraso_pago"] > 0).astype(int)).clip(0, 1).astype(int)
    df["incidencia_masiva_lag"] = df["incidencia_masiva_lag"].fillna(0).clip(0, 1).astype(int)
    df["stress_calidad_lag"] = df["stress_calidad_lag"].clip(0, 1)

    df = df.drop(columns=["tipo_plan_cliente", "num_lineas_cliente", "zona_id_cliente"])
    df = df.sort_values(["cliente_id", "fecha"]).reset_index(drop=True)
    return df, summarize(
        "facturacion_mensual",
        raw,
        df,
        ["cliente_id", "fecha"],
        "Se quitaron duplicados cliente-mes, se completaron tipo_plan/zona/num_lineas desde clientes y se reconstruyo importe_total cuando faltaba.",
    )


def clean_red() -> tuple[pd.DataFrame, QualityRecord]:
    raw = pd.read_csv(RAW_DIR / "calidad_senal_zona_mensual.csv")
    df = tidy_strings(raw).drop_duplicates()
    df["fecha"] = month_start(df["fecha"])

    num_cols = [
        "poblacion_zona",
        "cobertura_4g_pct",
        "cobertura_5g_pct",
        "latencia_ms",
        "velocidad_media_mbps",
        "tasa_cortes_pct",
        "indice_calidad_global",
        "incidencia_masiva",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = drop_best_record_per_key(df, ["zona_id", "fecha"])
    df = df.sort_values(["zona_id", "fecha"]).reset_index(drop=True)

    interpolate_cols = [
        "cobertura_4g_pct",
        "cobertura_5g_pct",
        "latencia_ms",
        "velocidad_media_mbps",
        "tasa_cortes_pct",
        "indice_calidad_global",
    ]
    for col in interpolate_cols:
        df[col] = df.groupby("zona_id", dropna=False)[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )

    df["cobertura_4g_pct"] = df["cobertura_4g_pct"].clip(0, 100)
    df["cobertura_5g_pct"] = df["cobertura_5g_pct"].clip(0, 100)
    df["tasa_cortes_pct"] = df["tasa_cortes_pct"].clip(0, 100)
    df["latencia_ms"] = df["latencia_ms"].clip(lower=0.1)
    df["velocidad_media_mbps"] = df["velocidad_media_mbps"].clip(lower=0.1)
    df["incidencia_masiva"] = df["incidencia_masiva"].fillna(0).clip(0, 1).astype(int)
    df["region"] = df["region"].fillna("Desconocido")
    df["tipo_zona"] = df["tipo_zona"].fillna("Desconocido")
    return df, summarize(
        "calidad_senal_zona_mensual",
        raw,
        df,
        ["zona_id", "fecha"],
        "Se deduplico zona-mes y se imputaron faltantes temporales por interpolacion dentro de cada zona.",
    )


def clean_encuestas() -> tuple[pd.DataFrame, QualityRecord]:
    raw = pd.read_csv(RAW_DIR / "encuestas_texto.csv")
    df = tidy_strings(raw).drop_duplicates()
    df["fecha"] = month_start(df["fecha"])

    for col in [
        "puntuacion_general_1a5",
        "nps_0a10",
        "indice_calidad_global",
        "incidencia_masiva",
        "stress_calidad",
        "flag_incongruente",
        "sent_text_latente",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = drop_best_record_per_key(df, ["encuesta_id"])
    df.loc[~df["puntuacion_general_1a5"].between(1, 5), "puntuacion_general_1a5"] = np.nan
    df.loc[~df["nps_0a10"].between(0, 10), "nps_0a10"] = np.nan
    df["incidencia_masiva"] = df["incidencia_masiva"].fillna(0).clip(0, 1).astype(int)
    df["flag_incongruente"] = df["flag_incongruente"].fillna(0).clip(0, 1).astype(int)
    df["stress_calidad"] = df["stress_calidad"].clip(0, 1)
    df["texto_libre"] = df["texto_libre"].fillna("")
    df["region"] = df["region"].fillna("Desconocido")
    df["tipo_zona"] = df["tipo_zona"].fillna("Desconocido")
    df = df.sort_values(["fecha", "encuesta_id"]).reset_index(drop=True)
    return df, summarize(
        "encuestas_texto",
        raw,
        df,
        ["encuesta_id"],
        "Se eliminaron encuestas repetidas y se invalidaron respuestas fuera de rango para que no sesguen agregados zonales.",
    )


def clean_soporte() -> tuple[pd.DataFrame, QualityRecord]:
    raw = pd.read_csv(RAW_DIR / "interacciones_soporte.csv")
    df = tidy_strings(raw).drop_duplicates()
    df["fecha_evento"] = pd.to_datetime(df["fecha_evento"], errors="coerce")
    df["mes"] = month_start(df["mes"])
    df["mes_evento"] = month_start(df["fecha_evento"])
    df["mes"] = df["mes"].fillna(df["mes_evento"])
    mismatch = df["mes"].notna() & df["mes_evento"].notna() & (df["mes"] != df["mes_evento"])
    df.loc[mismatch, "mes"] = df.loc[mismatch, "mes_evento"]

    for col in [
        "duracion_min",
        "resuelto",
        "satisfaccion_post",
        "stress_calidad_lag",
        "incidencia_masiva_lag",
        "impago_mes",
        "dias_retraso_mes",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = drop_best_record_per_key(df, ["interaccion_id"])
    df.loc[df["duracion_min"] <= 0, "duracion_min"] = np.nan
    df["duracion_min"] = impute_group_median(df, "duracion_min", "canal")
    df["satisfaccion_post"] = df["satisfaccion_post"].clip(1, 5)
    df["satisfaccion_post"] = impute_group_median(df, "satisfaccion_post", "canal")
    df["stress_calidad_lag"] = df["stress_calidad_lag"].clip(0, 1)
    df["resuelto"] = df["resuelto"].fillna(0).clip(0, 1).astype(int)
    df["incidencia_masiva_lag"] = df["incidencia_masiva_lag"].fillna(0).clip(0, 1).astype(int)
    df["impago_mes"] = df["impago_mes"].fillna(0).clip(0, 1).astype(int)
    df["dias_retraso_mes"] = df["dias_retraso_mes"].fillna(0).clip(lower=0)
    df["canal"] = df["canal"].fillna("desconocido")
    df["motivo"] = df["motivo"].fillna("Desconocido")
    df = df.drop(columns=["mes_evento"]).sort_values(["cliente_id", "mes", "interaccion_id"]).reset_index(drop=True)
    return df, summarize(
        "interacciones_soporte",
        raw,
        df,
        ["interaccion_id"],
        "Se consolidaron eventos unicos, se alineo el mes con la fecha del evento y se imputaron duraciones/satisfaccion por canal.",
    )


def write_outputs(cleaned: dict[str, pd.DataFrame], records: list[QualityRecord]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in cleaned.items():
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    report = pd.DataFrame([record.__dict__ for record in records])
    report.to_csv(PROCESSED_DIR / "data_quality_report.csv", index=False)

    lines = [
        "# Processed data",
        "",
        "Datasets limpiados y validados para exploracion y modelado.",
        "",
        "## Resumen",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"### {record.dataset}",
                f"- Filas raw: {record.rows_raw}",
                f"- Filas processed: {record.rows_clean}",
                f"- Filas eliminadas: {record.dropped_rows}",
                f"- Duplicados exactos raw: {record.exact_duplicates_raw}",
                f"- Duplicados por clave raw: {record.key_duplicates_raw}",
                f"- Duplicados por clave processed: {record.key_duplicates_clean}",
                f"- Nota: {record.notes}",
                "",
            ]
        )

    lines.extend(
        [
            "## Regeneracion",
            "",
            "Ejecuta:",
            "",
            "```bash",
            "python src/data/make_processed_data.py",
            "```",
            "",
            "Tambien se genera `data_quality_report.csv` con el detalle tabular.",
        ]
    )

    (PROCESSED_DIR / "readme.md").write_text("\n".join(lines), encoding="utf-8")
    (PROCESSED_DIR / "data_quality_report.json").write_text(
        json.dumps([record.__dict__ for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    churn_clean, churn_record = clean_churn_target()
    clientes_clean, clientes_record = clean_clientes()
    fact_clean, fact_record = clean_facturacion(clientes_clean)
    red_clean, red_record = clean_red()
    enc_clean, enc_record = clean_encuestas()
    soporte_clean, soporte_record = clean_soporte()

    cleaned = {
        "churn_target": churn_clean,
        "clientes": clientes_clean,
        "facturacion_mensual": fact_clean,
        "calidad_senal_zona_mensual": red_clean,
        "encuestas_texto": enc_clean,
        "interacciones_soporte": soporte_clean,
    }
    records = [
        churn_record,
        clientes_record,
        fact_record,
        red_record,
        enc_record,
        soporte_record,
    ]
    write_outputs(cleaned, records)
    print("Processed datasets generated in data/processed")


if __name__ == "__main__":
    main()
