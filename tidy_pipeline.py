from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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


# ---------------------------------------------------------------------------
#   TABELA PRINCIPAL DE COLISÕES
# ---------------------------------------------------------------------------

def build_collisions_table(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói a tabela principal de colisões com campos padronizados."""

    # Remover colunas *_outlier_orig
    df = df[[c for c in df.columns if not c.endswith("_outlier_orig")]]

    collisions_cols = [
        "collision_id",
        "crash_date_parsed",
        "crash_time_parsed",
        "borough_std",
        "zip_code_std",
        "latitude",
        "longitude",
        "on_street_name",
        "cross_street_name",
        "off_street_name",
        "number_of_persons_injured",
        "number_of_persons_killed",
    ]

    available_cols = [col for col in collisions_cols if col in df.columns]

    collisions = df[available_cols].copy()

    rename_map = {
        "crash_date_parsed": "crash_date",
        "crash_time_parsed": "crash_time",
        "borough_std": "borough",
        "zip_code_std": "zip_code",
    }
    collisions.rename(columns=rename_map, inplace=True)

    collisions["collision_id"] = collisions["collision_id"].astype(str)

    if "crash_date" in collisions.columns and "crash_time" in collisions.columns:
        collisions["crash_datetime"] = pd.to_datetime(
            collisions["crash_date"].astype(str)
            + " "
            + collisions["crash_time"].astype(str),
            errors="coerce",
        )
        collisions.drop(columns=["crash_time"], inplace=True)

    final_cols = [
        "collision_id",
        "crash_date",
        "crash_datetime",
        "borough",
        "zip_code",
        "latitude",
        "longitude",
        "on_street_name",
        "off_street_name",
        "cross_street_name",
        "number_of_persons_injured",
        "number_of_persons_killed",
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

    # Remove colunas com valores pré-winsorização
    df = df[[c for c in df.columns if not c.endswith("_outlier_orig")]]

    # Seleciona apenas colunas oficiais e reais
    count_cols = [
        col
        for col in df.columns
        if col.startswith("number_of_") and not col.endswith("_outlier_orig")
    ]

    id_vars = [
        col
        for col in ["collision_id", "crash_date_parsed", "crash_time_parsed", "borough_std"]
        if col in df.columns
    ]

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
            return int(float(val))
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

    rename_map = {
        "crash_date_parsed": "crash_date",
        "crash_time_parsed": "crash_time",
        "borough_std": "borough",
    }
    melted.rename(columns=rename_map, inplace=True)

    melted["collision_id"] = melted["collision_id"].astype(str)

    if "crash_date" in melted.columns and "crash_time" in melted.columns:
        melted["crash_datetime"] = pd.to_datetime(
            melted["crash_date"].astype(str)
            + " "
            + melted["crash_time"].astype(str),
            errors="coerce",
        )

    final_cols = [
        "collision_id",
        "crash_date",
        "crash_datetime",
        "borough",
        "participant_type",
        "injury_outcome",
        "people_count",
    ]
    final_cols = [c for c in final_cols if c in melted.columns]

    melted = melted[final_cols]
    return melted


# ---------------------------------------------------------------------------
#   VEÍCULOS
# ---------------------------------------------------------------------------

def build_vehicle_table(df: pd.DataFrame) -> pd.DataFrame:

    df = df[[c for c in df.columns if not c.endswith("_outlier_orig")]]

    available = _resolve_vehicle_columns(df)
    records: List[pd.DataFrame] = []
    base_cols = ["collision_id", "crash_date_parsed", "crash_time_parsed"]
    base_cols = [c for c in base_cols if c in df.columns]

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

    rename_map = {
        "crash_date_parsed": "crash_date",
        "crash_time_parsed": "crash_time",
    }
    vehicles.rename(columns=rename_map, inplace=True)

    if "crash_date" in vehicles.columns and "crash_time" in vehicles.columns:
        vehicles["crash_datetime"] = pd.to_datetime(
            vehicles["crash_date"].astype(str)
            + " "
            + vehicles["crash_time"].astype(str),
            errors="coerce",
        )

    final_cols = [
        "collision_id",
        "vehicle_index",
        "vehicle_type",
        "contributing_factor",
        "crash_date",
        "crash_datetime",
    ]
    final_cols = [c for c in final_cols if c in vehicles.columns]

    vehicles = vehicles[final_cols]
    vehicles = vehicles.dropna(
        subset=["vehicle_type", "contributing_factor"], how="all"
    )
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

    # Remover colunas *_outlier_orig de uma vez
    df = df[[c for c in df.columns if not c.endswith("_outlier_orig")]]

    collisions = build_collisions_table(df)
    participant_outcomes = build_participant_outcomes(df)
    # vehicles = build_vehicle_table(df)
    vehicles = pd.DataFrame()  # IGNORE

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
