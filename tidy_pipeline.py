"""Pipeline para transformação do dataset limpo em formato tidy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


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
        try:
            self.collisions.to_parquet(output_dir / "collisions.parquet", engine=engine, index=False)
            self.participant_outcomes.to_parquet(
                output_dir / "participant_outcomes.parquet", engine=engine, index=False
            )
            self.vehicles.to_parquet(output_dir / "vehicles.parquet", engine=engine, index=False)
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise ImportError(
                "Falha ao exportar para Parquet: instale 'pyarrow' ou 'fastparquet' no ambiente."
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


def build_collisions_table(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói a tabela principal de colisões com campos padronizados."""
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
    missing_cols = [col for col in collisions_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Colunas ausentes no dataset limpo: {missing_cols}")

    collisions = df[collisions_cols].copy()
    collisions.rename(
        columns={
            "crash_date_parsed": "crash_date",
            "crash_time_parsed": "crash_time",
            "borough_std": "borough",
            "zip_code_std": "zip_code",
        },
        inplace=True,
    )
    collisions["collision_id"] = collisions["collision_id"].astype(str)
    collisions["crash_datetime"] = pd.to_datetime(
        collisions["crash_date"].astype(str) + " " + collisions["crash_time"].astype(str),
        errors="coerce",
    )
    collisions.drop(columns=["crash_time"], inplace=True)
    collisions = collisions[
        [
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
    ]
    return collisions


def build_participant_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Gera tabela tidy com contagens de feridos/mortos por tipo de participante."""
    count_cols = [col for col in df.columns if col.startswith("number_of_")]
    id_vars = ["collision_id", "crash_date_parsed", "crash_time_parsed", "borough_std"]
    for col in id_vars:
        if col not in df.columns:
            raise KeyError(f"Coluna obrigatória ausente: {col}")
    melted = df[id_vars + count_cols].melt(
        id_vars=id_vars,
        value_vars=count_cols,
        var_name="metric",
        value_name="people_count",
    )
    def safe_int_conversion(val):
        if pd.isna(val):
            return 0
        if isinstance(val, str):
            if val.lower() in ['true', 'false']:
                return int(val.lower() == 'true')
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return 0
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0
    
    melted["people_count"] = melted["people_count"].apply(safe_int_conversion)

    def parse_metric(metric: str) -> Tuple[str, str]:
        prefix = "number_of_"
        if not metric.startswith(prefix):
            return ("unknown", metric)
        core = metric[len(prefix) :]
        parts = core.rsplit("_", 1)
        if len(parts) != 2:
            return ("unknown", core)
        participant_raw, outcome = parts
        participant_map = {
            "persons": "all_persons",
            "pedestrians": "pedestrian",
            "cyclist": "cyclist",
            "motorist": "motorist",
        }
        participant = participant_map.get(participant_raw, participant_raw)
        return participant, outcome

    parsed = melted["metric"].apply(parse_metric)
    melted["participant_type"] = parsed.apply(lambda x: x[0])
    melted["injury_outcome"] = parsed.apply(lambda x: x[1])
    melted.drop(columns=["metric"], inplace=True)
    melted.rename(
        columns={
            "crash_date_parsed": "crash_date",
            "crash_time_parsed": "crash_time",
            "borough_std": "borough",
        },
        inplace=True,
    )
    melted["collision_id"] = melted["collision_id"].astype(str)
    melted["crash_datetime"] = pd.to_datetime(
        melted["crash_date"].astype(str) + " " + melted["crash_time"].astype(str),
        errors="coerce",
    )
    melted = melted[
        [
            "collision_id",
            "crash_date",
            "crash_datetime",
            "borough",
            "participant_type",
            "injury_outcome",
            "people_count",
        ]
    ]
    return melted


def build_vehicle_table(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói tabela tidy com os veículos e fatores contribuintes."""
    available = _resolve_vehicle_columns(df)
    records: List[pd.DataFrame] = []
    base_cols = ["collision_id", "crash_date_parsed", "crash_time_parsed"]
    for col in base_cols:
        if col not in df.columns:
            raise KeyError(f"Coluna obrigatória ausente: {col}")
    for vehicle_index, vehicle_col, factor_col in available:
        temp = df[base_cols].copy()
        temp["collision_id"] = temp["collision_id"].astype(str)
        if vehicle_col in df.columns:
            temp["vehicle_type"] = df[vehicle_col]
        else:
            temp["vehicle_type"] = pd.NA
        if factor_col in df.columns:
            temp["contributing_factor"] = df[factor_col]
        else:
            temp["contributing_factor"] = pd.NA
        temp["vehicle_index"] = vehicle_index
        records.append(temp)
    vehicles = pd.concat(records, ignore_index=True)
    vehicles.rename(
        columns={
            "crash_date_parsed": "crash_date",
            "crash_time_parsed": "crash_time",
        },
        inplace=True,
    )
    vehicles["crash_datetime"] = pd.to_datetime(
        vehicles["crash_date"].astype(str) + " " + vehicles["crash_time"].astype(str),
        errors="coerce",
    )
    vehicles = vehicles[
        [
            "collision_id",
            "vehicle_index",
            "vehicle_type",
            "contributing_factor",
            "crash_date",
            "crash_datetime",
        ]
    ]
    vehicles = vehicles.dropna(subset=["vehicle_type", "contributing_factor"], how="all")
    vehicles.reset_index(drop=True, inplace=True)
    return vehicles


def run_pipeline(
    cleaned_csv_path: Path | str = Path("cleaned_data.csv"),
    output_dir: Path | str = Path("data") / "tidy",
    *,
    export_parquet: bool = True,
    parquet_engine: str | None = "pyarrow",
) -> TidyDataArtifacts:
    """Executa a pipeline tidy completa."""
    cleaned_csv_path = Path(cleaned_csv_path)
    if not cleaned_csv_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {cleaned_csv_path}")

    df = pd.read_csv(cleaned_csv_path)
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