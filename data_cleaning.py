from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

ENDPOINT = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"

###############################################################################
# DOWNLOAD
###############################################################################

def download_data(limit: int = 1000) -> list:
    params = {"$limit": str(limit)}
    resp = requests.get(ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

###############################################################################
# NORMALIZAÇÃO DE COLUNAS
###############################################################################

def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )
    return df

###############################################################################
# PRE-FILTRO DE VALORES ABSURDOS
###############################################################################
# NYC JAMAIS registrou:
# - > 50 mortos em um acidente
# - > 200 feridos em um acidente
# Valores acima disso indicam erro de parsing ou lixo histórico

def remove_impossible_values(df: pd.DataFrame, log: dict):
    count_cols = [c for c in df.columns if "injured" in c or "killed" in c]

    impossible_mask = pd.DataFrame(False, index=df.index, columns=count_cols)
    for c in count_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

        # limite seguro baseado em estatística real
        mask = (df[c] > 200)
        impossible_mask[c] = mask
        df.loc[mask, c] = np.nan

    log["removed_impossible_values"] = {
        c: int(impossible_mask[c].sum()) for c in count_cols
    }

    return df

###############################################################################
# CORREÇÃO DE INCONSISTÊNCIAS ENTRE TOTAIS E COMPONENTES
###############################################################################

def fix_inconsistencies(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    df = df.copy()
    corrections = defaultdict(int)

    # INJURED
    comp_inj_cols = [
        c for c in df.columns 
        if c.endswith("_injured") and c != "number_of_persons_injured"
    ]

    if "number_of_persons_injured" in df.columns:
        comp_sum = df[comp_inj_cols].fillna(0).sum(axis=1)

        total = df["number_of_persons_injured"].fillna(0)

        mask = comp_sum > total
        corrections["injured_rows_fixed"] = int(mask.sum())

        # 🔥 FIX: garantir dtype correto
        df.loc[mask, "number_of_persons_injured"] = (
            comp_sum[mask]
            .astype("Int64")
        )

    # KILLED
    comp_kill_cols = [
        c for c in df.columns 
        if c.endswith("_killed") and c != "number_of_persons_killed"
    ]

    if "number_of_persons_killed" in df.columns:
        comp_sum = df[comp_kill_cols].fillna(0).sum(axis=1)
        total = df["number_of_persons_killed"].fillna(0)

        mask = comp_sum > total
        corrections["killed_rows_fixed"] = int(mask.sum())

        df.loc[mask, "number_of_persons_killed"] = (
            comp_sum[mask]
            .astype("Int64")
        )

    log["inconsistency_fixes"] = corrections
    return df


###############################################################################
# OUTLIERS ROBUSTOS (winsorização corrigida)
###############################################################################

def robust_winsorize(df: pd.DataFrame, count_cols: list, log: dict):
    outlier_info = {}

    for c in count_cols:
        ser = df[c].dropna()

        if ser.empty:
            outlier_info[c] = {"n_outliers": 0}
            continue

        # percentis sólidos (1–99%)
        p1 = np.percentile(ser, 1)
        p99 = np.percentile(ser, 99)

        mask = (df[c] < p1) | (df[c] > p99)
        n_out = int(mask.sum())

        df[c] = df[c].clip(lower=p1, upper=p99)

        outlier_info[c] = {
            "p1": float(p1),
            "p99": float(p99),
            "n_outliers": n_out,
        }

    log["winsorized"] = outlier_info
    return df

###############################################################################
# LIMPEZA COMPLETA
###############################################################################

def main(limit: int = 5000, out_prefix: str = "cleaned"):
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "limit": limit,
        "steps": []
    }

    # DOWNLOAD
    raw = download_data(limit)
    df = pd.DataFrame(raw)

    # NORMALIZAR COLUNAS
    df = normalize_colnames(df)

    # DETECTAR COLUNAS DE CONTAGEM
    count_cols = [c for c in df.columns if "injured" in c or "killed" in c]
    log["count_columns"] = count_cols

    # TORNAR NUMÉRICO
    for c in count_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1️⃣ REMOVER VALORES IMPOSSÍVEIS
    df = remove_impossible_values(df, log)

    # 2️⃣ CORRIGIR INCONSISTÊNCIAS ENTRE TOTAIS E COMPONENTES
    df = fix_inconsistencies(df, log)

    # 3️⃣ APLICAR WINSORIZAÇÃO ROBUSTA
    df = robust_winsorize(df, count_cols, log)

    # 4️⃣ PARSE DA DATA
    if "crash_date" in df.columns:
        df["crash_date_parsed"] = pd.to_datetime(df["crash_date"], errors="coerce")

    # SALVAR
    df.to_csv(
        f"{out_prefix}.csv",
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        quotechar='"',
        line_terminator="\n",
        on_bad_lines='skip'
    )

    with open(f"{out_prefix}_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print("✔ Cleaning finalizado!")
    print(f"Arquivo salvo: {out_prefix}.csv")

###############################################################################
# Funções utilitárias
###############################################################################

def is_count_col(col: str) -> bool:
    """Compatibilidade com o main.ipynb."""
    return (
        col.startswith("number_of_")
        or "injured" in col
        or "killed" in col
    )


def standardize_zip(z):
    """Compatibilidade — retorna zip padronizado."""
    if pd.isna(z):
        return None
    z = str(z).strip()
    digits = ''.join(ch for ch in z if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(5)[:5]


def missing_counts_df(df: pd.DataFrame):
    """Retorna um DF com contagem de missing por coluna."""
    return df.isna().sum().reset_index().rename(
        columns={"index": "column", 0: "missing_count"}
    )


def check_inconsistencies_df(df: pd.DataFrame):
    """Verifica linhas onde soma dos componentes > total."""
    issues = []

    inj_cols = [c for c in df.columns if c.endswith("_injured") and c != "number_of_persons_injured"]
    kill_cols = [c for c in df.columns if c.endswith("_killed") and c != "number_of_persons_killed"]

    if inj_cols:
        comp_sum = df[inj_cols].sum(axis=1)
        mask = comp_sum > df["number_of_persons_injured"]
        issues.append(("injured", int(mask.sum())))

    if kill_cols:
        comp_sum = df[kill_cols].sum(axis=1)
        mask = comp_sum > df["number_of_persons_killed"]
        issues.append(("killed", int(mask.sum())))

    return pd.DataFrame(issues, columns=["type", "rows_with_inconsistency"])


def plot_missing_heatmap(df: pd.DataFrame, figsize=(14,6)):
    """Plota gráfico de missing (versão simplificada)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=figsize)
    sns.heatmap(df.isna(), cbar=False)
    plt.title("Missing Heatmap")
    plt.show()


###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--out", type=str, default="cleaned")
    args = parser.parse_args()

    main(limit=args.limit, out_prefix=args.out)
