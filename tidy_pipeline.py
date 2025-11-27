from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re

import pandas as pd
import json


@dataclass
class TidyDataArtifacts:
    """Container para os DataFrames produzidos pela pipeline tidy."""

    collisions: pd.DataFrame
    participant_outcomes: pd.DataFrame
    vehicles: pd.DataFrame

    def to_parquet(self, output_dir: Path, *, engine: str | None = "pyarrow") -> None:
        """Salva os artefatos tidy em arquivos parquet no diretório informado."""
        if engine is None:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        def _clean_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
            """Converte colunas object para string nullable de forma segura para PyArrow."""
            df = df.copy()

            def _normalize_value(v):
                if pd.isna(v):
                    return None
                if isinstance(v, (bytes, bytearray)):
                    try:
                        return v.decode("utf-8", errors="replace")
                    except Exception:
                        return str(v)
                if isinstance(v, (list, dict)):
                    try:
                        return json.dumps(v, ensure_ascii=False)
                    except Exception:
                        return str(v)
                return v

            for col in df.columns:
                series = df[col]
                if (
                    pd.api.types.is_object_dtype(series)
                    or pd.api.types.is_string_dtype(series)
                    or pd.api.types.is_categorical_dtype(series)
                ):
                    try:
                        series = series.map(_normalize_value)
                        series = series.replace({"nan": None, "NaN": None})
                        df[col] = series.astype("string")
                    except Exception:
                        df[col] = (
                            series.astype(str)
                            .replace({"nan": None, "NaN": None})
                            .astype("string")
                        )

            return df

        try:
            collisions_clean = _clean_for_parquet(self.collisions)
            participant_outcomes_clean = _clean_for_parquet(self.participant_outcomes)
            vehicles_clean = _clean_for_parquet(self.vehicles)

            collisions_clean.to_parquet(
                output_dir / "collisions.parquet", engine=engine, index=False
            )
            participant_outcomes_clean.to_parquet(
                output_dir / "participant_outcomes.parquet",
                engine=engine,
                index=False,
            )
            vehicles_clean.to_parquet(
                output_dir / "vehicles.parquet", engine=engine, index=False
            )

        except ImportError as exc:
            raise ImportError(
                "Falha ao exportar para Parquet: instale 'pyarrow' ou 'fastparquet' no ambiente."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Erro ao escrever arquivos Parquet: {exc}\n"
                "Verifique tipos de dados e valores nas colunas."
            ) from exc


def _resolve_vehicle_columns(df: pd.DataFrame) -> List[Tuple[int, str, str]]:
    """Mapeia o índice do veículo para os nomes de colunas existentes."""
    vehicle_cols: Dict[int, str] = {
        1: "vehicle_type_code1",
        2: "vehicle_type_code2",
        3: "vehicle_type_code_3",
        4: "vehicle_type_code_4",
        5: "vehicle_type_code_5",
    }
    factor_cols: Dict[int, str] = {
        1: "contributing_factor_vehicle_1",
        2: "contributing_factor_vehicle_2",
        3: "contributing_factor_vehicle_3",
        4: "contributing_factor_vehicle_4",
        5: "contributing_factor_vehicle_5",
    }
    available: List[Tuple[int, str, str]] = []
    for idx in vehicle_cols:
        if vehicle_cols[idx] in df.columns or factor_cols[idx] in df.columns:
            available.append((idx, vehicle_cols.get(idx, ""), factor_cols.get(idx, "")))
    return available


def _strip_outlier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas auxiliares criadas na winsorização."""
    return df[[c for c in df.columns if not c.endswith("_outlier_orig")]]


def _clean_collision_id(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza collision_id e remove registros inválidos."""
    df = df.copy()
    df["collision_id"] = df["collision_id"].astype(str)
    valid = df["collision_id"].str.fullmatch(r"[1-9][0-9]*")
    return df[valid].copy()


def _sanitize_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Torna colunas de contagem numéricas e limita valores extremos."""
    df = df.copy()
    count_cols = [
        col
        for col in df.columns
        if col.startswith("number_of_")
        and (col.endswith("injured") or col.endswith("killed"))
    ]
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        limit = 50 if col.endswith("killed") else 200
        bad = (df[col] < 0) | (df[col] > limit)
        df.loc[bad, col] = pd.NA
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas derivadas de data/hora para análises SQL e visuais."""
    df = df.copy()

    crash_date_series = df.get("crash_date_parsed", df.get("crash_date"))
    crash_time_series = df.get("crash_time_parsed", df.get("crash_time"))

    df["crash_date"] = pd.to_datetime(crash_date_series, errors="coerce")
    df["crash_time"] = crash_time_series

    df["crash_datetime"] = pd.to_datetime(
        df["crash_date"].astype(str) + " " + df["crash_time"].astype(str),
        errors="coerce",
    )
    df["crash_year"] = df["crash_datetime"].dt.year
    df["crash_month"] = df["crash_datetime"].dt.month
    df["crash_hour"] = df["crash_datetime"].dt.hour
    df["crash_dow"] = df["crash_datetime"].dt.day_name()

    def _period(hour: float) -> str:
        if pd.isna(hour):
            return "unknown"
        hour = int(hour)
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 18:
            return "afternoon"
        if 18 <= hour < 24:
            return "evening"
        return "night"

    df["period_of_day"] = df["crash_hour"].apply(_period)
    return df


def _clean_contributing_factor(factor: object, vehicle_type: object) -> str | None:
    """Sanitiza fator contribuinte removendo lixos (nulos, numéricos, ruas, tipos de veículo)."""
    if pd.isna(factor):
        return None

    raw = str(factor).strip()
    if raw == "":
        return None

    low = raw.lower()
    if low in {"unspecified", "unknown", "nan", "none", "null", "0", "na", "n/a"}:
        return None

    # remove registros que são apenas números
    if re.fullmatch(r"[0-9]+(?:\\.[0-9]+)?", raw):
        return None

    # descarta valores que parecem nomes de via
    upper = raw.upper()
    street_tokens = [" AVENUE", " AVE", " STREET", " ST ", " BROADWAY", " HIGHWAY", " HWY", " RD", " ROAD"]
    if any(tok in upper for tok in street_tokens):
        return None

    # se o fator for igual ao tipo de veículo, provavelmente é ruído
    if vehicle_type is not None:
        vt = str(vehicle_type).strip().lower()
        if vt and vt == low:
            return None

    # descarta fatores que são na verdade tipo de veículo
    vehicle_tokens = {
        "sedan",
        "taxi",
        "station wagon",
        "sport utility vehicle",
        "suv",
        "van",
        "minivan",
        "pickup",
        "pick-up",
        "truck",
        "box truck",
        "flat bed",
        "bus",
        "school bus",
        "bike",
        "bicycle",
        "motorcycle",
        "scooter",
    }
    if low in vehicle_tokens:
        return None
    if any(tok in low for tok in vehicle_tokens):
        return None

    return raw


# ---------------------------------------------------------------------------
#   TABELA PRINCIPAL DE COLISÕES
# ---------------------------------------------------------------------------

def build_collisions_table(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói a tabela principal de colisões com campos padronizados."""

    # Remover colunas *_outlier_orig
    df = df[[c for c in df.columns if not c.endswith("_outlier_orig")]]

    collisions_cols = [
        "collision_id",
        "crash_date",
        "crash_datetime",
        "crash_year",
        "crash_month",
        "crash_hour",
        "period_of_day",
        "borough",
        "borough_std",
        "zip_code",
        "zip_code_std",
        "latitude",
        "longitude",
        "on_street_name",
        "cross_street_name",
        "off_street_name",
        "number_of_persons_injured",
        "number_of_persons_killed",
        "number_of_pedestrians_injured",
        "number_of_pedestrians_killed",
        "number_of_cyclist_injured",
        "number_of_cyclist_killed",
        "number_of_motorist_injured",
        "number_of_motorist_killed",
    ]

    available_cols = [col for col in collisions_cols if col in df.columns]
    collisions = df[available_cols].copy()

    if "borough_std" in collisions.columns:
        collisions["borough"] = collisions.pop("borough_std")
    if "zip_code_std" in collisions.columns:
        collisions["zip_code"] = collisions.pop("zip_code_std")

    collisions["collision_id"] = collisions["collision_id"].astype(str)
    collisions["is_fatal"] = collisions["number_of_persons_killed"].fillna(0) > 0
    collisions["people_injured_total"] = collisions["number_of_persons_injured"]
    collisions["people_killed_total"] = collisions["number_of_persons_killed"]

    final_cols = [
        "collision_id",
        "crash_date",
        "crash_datetime",
        "crash_year",
        "crash_month",
        "crash_hour",
        "period_of_day",
        "borough",
        "zip_code",
        "latitude",
        "longitude",
        "on_street_name",
        "off_street_name",
        "cross_street_name",
        "number_of_persons_injured",
        "number_of_persons_killed",
        "number_of_pedestrians_injured",
        "number_of_pedestrians_killed",
        "number_of_cyclist_injured",
        "number_of_cyclist_killed",
        "number_of_motorist_injured",
        "number_of_motorist_killed",
        "is_fatal",
        "people_injured_total",
        "people_killed_total",
    ]
    final_cols = [c for c in final_cols if c in collisions.columns]
    collisions = collisions[final_cols]
    return collisions


# ---------------------------------------------------------------------------
#   PARTICIPANT OUTCOMES — TIDY
#   Corrigido para evitar colunas *_outlier_orig
# ---------------------------------------------------------------------------

def build_participant_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Gera tabela tidy com contagens de feridos/mortos por tipo de participante."""

    df = _strip_outlier_columns(df)

    count_cols = [
        col
        for col in df.columns
        if col.startswith("number_of_") and not col.endswith("_is_outlier")
    ]

    id_candidates = [
        "collision_id",
        "crash_date",
        "crash_datetime",
        "crash_year",
        "crash_month",
        "crash_hour",
        "period_of_day",
        "borough_std",
        "borough",
    ]
    id_vars = [col for col in id_candidates if col in df.columns]
    if "borough_std" in id_vars and "borough" in id_vars:
        id_vars.remove("borough")

    melted = df[id_vars + count_cols].melt(
        id_vars=id_vars,
        value_vars=count_cols,
        var_name="metric",
        value_name="people_count",
    )

    def safe_int(val):
        if pd.isna(val):
            return 0
        try:
            return max(int(float(val)), 0)
        except Exception:
            return 0

    melted["people_count"] = melted["people_count"].apply(safe_int)

    def parse_metric(metric: str):
        prefix = "number_of_"
        core = metric[len(prefix):] if metric.startswith(prefix) else metric
        parts = core.rsplit("_", 1)

        if len(parts) != 2:
            return ("unknown", core)

        participant_raw, outcome = parts

        participant_map = {
            "persons": "all_persons",
            "pedestrians": "pedestrian",
            "pedestrian": "pedestrian",
            "cyclist": "cyclist",
            "motorist": "motorist",
        }

        participant = participant_map.get(participant_raw, participant_raw)
        return participant, outcome

    parsed = melted["metric"].apply(parse_metric)
    melted["participant_type"] = parsed.apply(lambda x: x[0])
    melted["injury_outcome"] = parsed.apply(lambda x: x[1])
    melted.drop(columns=["metric"], inplace=True)

    melted["collision_id"] = melted["collision_id"].astype(str)
    melted = melted.rename(columns={"borough_std": "borough"})

    final_cols = [
        "collision_id",
        "crash_date",
        "crash_datetime",
        "crash_year",
        "crash_month",
        "crash_hour",
        "period_of_day",
        "borough",
        "participant_type",
        "injury_outcome",
        "people_count",
    ]
    final_cols = [c for c in final_cols if c in melted.columns]

    melted = melted[melted["participant_type"] != "all_persons"]

    melted = (
        melted.groupby(
            [
                "collision_id",
                "participant_type",
                "injury_outcome",
                "crash_datetime",
                "crash_year",
                "crash_month",
                "crash_hour",
                "period_of_day",
                "borough",
            ],
            as_index=False,
        )["people_count"]
        .max()
    )

    return melted


# ---------------------------------------------------------------------------
#   VEÍCULOS
# ---------------------------------------------------------------------------

def build_vehicle_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza informações de veículos e fatores contribuintes."""
    df = _strip_outlier_columns(df)

    available = _resolve_vehicle_columns(df)
    records: List[pd.DataFrame] = []
    base_cols = [
        "collision_id",
        "crash_date",
        "crash_datetime",
        "crash_year",
        "crash_month",
        "crash_hour",
        "period_of_day",
        "borough",
        "borough_std",
    ]
    base_cols = [c for c in base_cols if c in df.columns]
    if "borough_std" in base_cols and "borough" in base_cols:
        base_cols.remove("borough")

    for idx, vehicle_col, factor_col in available:
        temp = df[base_cols].copy()
        temp["collision_id"] = temp["collision_id"].astype(str)

        temp["vehicle_type"] = df.get(vehicle_col, pd.NA)
        temp["contributing_factor"] = df.get(factor_col, pd.NA)
        temp["vehicle_index"] = idx
        records.append(temp)

    if not records:
        return pd.DataFrame()

    vehicles = pd.concat(records, ignore_index=True)
    vehicles.rename(columns={"borough_std": "borough"}, inplace=True)

    # limpa fatores inválidos (numéricos, vias, iguais ao tipo de veículo, nulos)
    vehicles["contributing_factor"] = vehicles.apply(
        lambda row: _clean_contributing_factor(row["contributing_factor"], row.get("vehicle_type")),
        axis=1,
    )

    vehicles = vehicles.dropna(subset=["contributing_factor"], how="any")
    vehicles = vehicles.drop_duplicates()
    vehicles.reset_index(drop=True, inplace=True)
    return vehicles


# ---------------------------------------------------------------------------
#   EXECUÇÃO COMPLETA DA PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(
    cleaned_csv_path: Path | str = Path("cleaned_data.csv"),
    output_dir: Path | str = Path("data") / "tidy",
    *,
    export_parquet: bool = True,
    parquet_engine: str | None = "pyarrow",
) -> TidyDataArtifacts:

    cleaned_csv_path = Path(cleaned_csv_path)
    if not cleaned_csv_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {cleaned_csv_path}")

    df = pd.read_csv(cleaned_csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    df = _clean_collision_id(df)
    df = _strip_outlier_columns(df)
    df = _sanitize_counts(df)
    df = _add_temporal_features(df)

    collisions = build_collisions_table(df)
    participant_outcomes = build_participant_outcomes(df)
    vehicles = build_vehicle_table(df)

    artifacts = TidyDataArtifacts(
        collisions=collisions,
        participant_outcomes=participant_outcomes,
        vehicles=vehicles,
    )

    if export_parquet:
        artifacts.to_parquet(Path(output_dir), engine=parquet_engine)

    return artifacts


if __name__ == "__main__":
    try:
        artifacts = run_pipeline()
        print("Tabelas tidy salvas em data/tidy")
    except ImportError as exc:
        print(exc)
