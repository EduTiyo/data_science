"""
Data cleaning script for NYC motor vehicle collisions dataset
Endpoint: https://data.cityofnewyork.us/resource/h9gi-nx95.json

This script downloads a sample (configurable), inspects schema and basic stats,
applies cleaning transformations (missing values, outliers, inconsistencies,
padronization), and writes cleaned CSV + a JSON log describing all decisions.

Run: import data_cleaning; data_cleaning.main(limit=5000, out_prefix="cleaned_data")
"""
from __future__ import annotations
import json
import math
import os
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import requests

ENDPOINT = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"

def download_data(limit: int = 1000) -> list:
    params = {"$limit": str(limit)}
    resp = requests.get(ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    newcols = {}
    for c in df.columns:
        nc = c.strip().lower().replace(" ", "_").replace("/", "_")
        newcols[c] = nc
    df.rename(columns=newcols, inplace=True)
    return df

def is_count_col(col: str) -> bool:
    # heuristic: many count fields start with number_of_
    return col.startswith("number_of_") or any(k in col for k in ("injured", "killed"))

def standardize_zip(z):
    if pd.isna(z):
        return None
    z = str(z).strip()
    digits = ''.join(ch for ch in z if ch.isdigit())
    if len(digits) == 0:
        return None
    if len(digits) > 5:
        digits = digits[:5]
    return digits.zfill(5)

def detect_and_treat_outliers(df: pd.DataFrame, cols: list, log: dict):
    # flag outliers using IQR and then winsorize at 1st/99th percentiles
    outlier_info = {}
    for c in cols:
        ser = df[c].dropna().astype(float)
        if ser.empty:
            outlier_info[c] = {"n_outliers": 0}
            continue
        q1 = ser.quantile(0.25)
        q3 = ser.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[c] < lower) | (df[c] > upper)][c]
        n_out = int(outliers.shape[0])
        outlier_info[c] = {"n_outliers": n_out, "lower": float(lower), "upper": float(upper)}
        p1 = ser.quantile(0.01)
        p99 = ser.quantile(0.99)

        # Se os percentis forem iguais (ex.: p1 == p99 == 0 em colunas raras como fatalities),
        # a winsorização achataria toda a série para um único valor.
        # Nesse caso, apenas registramos no log e pulamos a etapa.
        if p99 <= p1:
            outlier_info[c].update(
                {
                    "winsor_lower": float(p1),
                    "winsor_upper": float(p99),
                    "skipped": "quantile_range_zero",
                }
            )
            continue

        df[c] = df[c].clip(lower=p1, upper=p99)
        outlier_info[c].update({"winsor_lower": float(p1), "winsor_upper": float(p99)})
    log['outliers'] = outlier_info
    return df

def fix_inconsistencies(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    # For injured/killed totals vs components, adjust total to component sums if component sum > total
    df = df.copy()
    corrections = defaultdict(int)
    comp_inj_cols = [c for c in df.columns if c.endswith("_injured") and c != 'number_of_persons_injured']
    if 'number_of_persons_injured' in df.columns and comp_inj_cols:
        comp_sum = df[comp_inj_cols].fillna(0).sum(axis=1)
        total = pd.to_numeric(df['number_of_persons_injured'], errors='coerce').fillna(0)
        mask = comp_sum > total
        corrections['injured_rows_fixed'] = int(mask.sum())
        df.loc[mask, 'number_of_persons_injured'] = comp_sum[mask]
    comp_kill_cols = [c for c in df.columns if c.endswith("_killed") and c != 'number_of_persons_killed']
    if 'number_of_persons_killed' in df.columns and comp_kill_cols:
        comp_sum = df[comp_kill_cols].fillna(0).sum(axis=1)
        total = pd.to_numeric(df['number_of_persons_killed'], errors='coerce').fillna(0)
        mask = comp_sum > total
        corrections['killed_rows_fixed'] = int(mask.sum())
        df.loc[mask, 'number_of_persons_killed'] = comp_sum[mask]
    log['inconsistency_fixes'] = corrections
    return df

def missing_counts_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with missing counts per column.

    Counts both NaN and empty-string as missing (consistent with notebook code).
    """
    rows = []
    for c in df.columns:
        cnt = int(df[c].replace('', pd.NA).isna().sum())
        rows.append({'column': c, 'missing': cnt})
    return pd.DataFrame(rows)

def check_inconsistencies_df(df: pd.DataFrame) -> dict:
    """Check inconsistencies where sum of component injured/killed > total.

    Returns a dict like {'injured': n, 'killed': n} with counts of violating rows.
    """
    dfn = df.copy()
    out = {}
    for c in dfn.columns:
        try:
            dfn[c] = pd.to_numeric(dfn[c], errors='coerce')
        except Exception:
            pass
    if 'number_of_persons_injured' in dfn.columns:
        comp_inj = [c for c in dfn.columns if c.endswith('_injured') and c != 'number_of_persons_injured']
        if comp_inj:
            comp_sum = dfn[comp_inj].fillna(0).sum(axis=1)
            total = dfn['number_of_persons_injured'].fillna(0)
            mask = comp_sum > total
            out['injured'] = int(mask.sum())
    if 'number_of_persons_killed' in dfn.columns:
        comp_kill = [c for c in dfn.columns if c.endswith('_killed') and c != 'number_of_persons_killed']
        if comp_kill:
            comp_sum_k = dfn[comp_kill].fillna(0).sum(axis=1)
            total_k = dfn['number_of_persons_killed'].fillna(0)
            maskk = comp_sum_k > total_k
            out['killed'] = int(maskk.sum())
    return out

def plot_missing_heatmap(df: pd.DataFrame, title: str, sample_n: int = 2000):
    """Plot a heatmap of missingness (sampled) using seaborn/matplotlib.

    This function avoids heavy memory usage by sampling when df is large.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    df2 = df.replace('', np.nan)
    if len(df2) > sample_n:
        df2 = df2.sample(sample_n, random_state=0)
    m = df2.isnull().astype(int)
    plt.figure(figsize=(14, max(6, 0.25 * len(m.columns))))
    sns.heatmap(m.T, cbar=False, cmap='viridis')
    plt.title(title)
    plt.xlabel('amostra de linhas')
    plt.ylabel('colunas')
    plt.tight_layout()
    plt.show()

def clean_dataframe(raw_df: pd.DataFrame, out_prefix: str = "cleaned_data", save: bool = True) -> pd.DataFrame:
    """
    Aplica a pipeline de limpeza ao DataFrame raw_df (já carregado).
    Se save=True, grava <out_prefix>.csv e <out_prefix>_cleaning_log.json.
    Retorna o DataFrame limpo.
    """
    log = {
        'run_timestamp': datetime.utcnow().isoformat() + 'Z',
        'source': 'in-memory',
        'decisions': []
    }

    df = raw_df.copy()
    log['start_rows'] = int(df.shape[0])

    # Normalizar nomes
    df = normalize_colnames(df)
    log['decisions'].append("Normalized column names: strip, lower, spaces->underscore")

    # Detectar colunas de contagem e converter/imputar
    count_cols = [c for c in df.columns if is_count_col(c)]
    log['count_cols_detected'] = count_cols
    for c in count_cols:
        df[c] = pd.to_numeric(df[c].replace('', np.nan), errors='coerce')
        n_missing_before = int(df[c].isna().sum())
        df[c] = df[c].fillna(0).astype(int)
        log['decisions'].append(f"Column {c}: coerced to int; missing({n_missing_before}) -> filled with 0")

    # Datas
    if 'crash_date' in df.columns:
        df['crash_date_parsed'] = pd.to_datetime(df['crash_date'], errors='coerce').dt.date
        log['decisions'].append(f"Parsed crash_date; missing_after={int(df['crash_date_parsed'].isna().sum())}")

    # Horários (simples)
    if 'crash_time' in df.columns:
        def _parse_time(x):
            try:
                if pd.isna(x) or str(x).strip() == '':
                    return pd.NaT
                t = pd.to_datetime(str(x).strip(), errors='coerce')
                return t.time() if not pd.isna(t) else pd.NaT
            except Exception:
                return pd.NaT
        df['crash_time_parsed'] = df['crash_time'].apply(_parse_time)
        log['decisions'].append(f"Parsed crash_time; missing_after={int(df['crash_time_parsed'].isna().sum())}")

    # Coordenadas
    for loc in ('latitude', 'longitude'):
        if loc in df.columns:
            df[loc] = pd.to_numeric(df[loc].replace('', np.nan), errors='coerce')
            df.loc[df[loc] == 0.0, loc] = np.nan
            log['decisions'].append(f"Column {loc}: coerced to float; zeros->NaN")

    # ZIP / borough
    if 'zip_code' in df.columns:
        df['zip_code_std'] = df['zip_code'].apply(standardize_zip)
        log['decisions'].append("Standardized zip_code -> zip_code_std")
    if 'borough' in df.columns:
        df['borough_std'] = df['borough'].astype(str).str.strip().str.upper().replace({'NONE': None, '': None})
        log['decisions'].append("Standardized borough -> borough_std")

    # Inconsistências e outliers (usa funções já no módulo)
    df = fix_inconsistencies(df, log)
    df = detect_and_treat_outliers(df, count_cols, log)

    log['rows_after_clean'] = int(df.shape[0])
    log['missing_pct_after'] = ((df.isna() | (df == '')).mean()).to_dict()

    if save:
        out_csv = f"{out_prefix}.csv"
        out_log = f"{out_prefix}_cleaning_log.json"
        df.to_csv(out_csv, index=False)
        with open(out_log, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

    return df

def main(limit: int = 5000, out_prefix: str = "cleaned_data"):
    log = {
        'run_timestamp': datetime.utcnow().isoformat() + 'Z',
        'source_endpoint': ENDPOINT,
        'limit_requested': int(limit),
        'decisions': []
    }
    print(f"Downloading {limit} rows from endpoint...")
    raw = download_data(limit)
    print(f"Downloaded {len(raw)} records")
    df = pd.DataFrame(raw)
    start_rows = df.shape[0]
    log['start_rows'] = int(start_rows)
    df = normalize_colnames(df)
    log['decisions'].append("Normalized column names: strip, lower, spaces->underscore")
    missing_pct = (df.isna() | (df == '')).mean().to_dict()
    log['missing_pct_before'] = {k: float(v) for k, v in missing_pct.items()}
    count_cols = [c for c in df.columns if is_count_col(c)]
    log['count_cols_detected'] = count_cols
    for c in count_cols:
        df[c] = pd.to_numeric(df[c].replace('', np.nan), errors='coerce')
        n_missing_before = int(df[c].isna().sum())
        df[c] = df[c].fillna(0).astype(int)
        log['decisions'].append(f"Column {c}: coerced to int; missing({n_missing_before}) -> filled with 0")
    if 'crash_date' in df.columns:
        df['crash_date_parsed'] = pd.to_datetime(df['crash_date'], errors='coerce').dt.date
        n_date_missing = int(df['crash_date_parsed'].isna().sum())
        log['decisions'].append(f"Parsed crash_date -> crash_date_parsed; missing after parse: {n_date_missing}")
    if 'crash_time' in df.columns:
        def parse_time(x):
            try:
                if pd.isna(x) or str(x).strip() == '':
                    return None
                t = pd.to_datetime(str(x).strip(), format='%H:%M', errors='coerce')
                if pd.isna(t):
                    t = pd.to_datetime(str(x).strip(), format='%H:%M:%S', errors='coerce')
                if pd.isna(t):
                    t = pd.to_datetime(str(x).strip(), errors='coerce')
                return t.time() if not pd.isna(t) else None
            except Exception:
                return None
        df['crash_time_parsed'] = df['crash_time'].apply(parse_time)
        n_time_missing = int(df['crash_time_parsed'].isna().sum())
        log['decisions'].append(f"Parsed crash_time -> crash_time_parsed; missing after parse: {n_time_missing}")
    for loc in ('latitude', 'longitude'):
        if loc in df.columns:
            df[loc] = pd.to_numeric(df[loc].replace('', np.nan), errors='coerce')
            mask0 = df[loc].astype(float) == 0.0
            n0 = int(mask0.sum())
            df.loc[mask0, loc] = np.nan
            log['decisions'].append(f"Column {loc}: coerced to float; replaced {n0} zero values with NaN")
    if 'zip_code' in df.columns:
        df['zip_code_std'] = df['zip_code'].apply(standardize_zip)
        n_zip_bad = int(df['zip_code_std'].isna().sum())
        log['decisions'].append(f"Standardized zip_code -> zip_code_std; invalid/empty: {n_zip_bad}")
    if 'borough' in df.columns:
        df['borough_std'] = df['borough'].astype(str).str.strip().str.upper().replace({'NONE': None, '': None})
        log['decisions'].append("Standardized borough to uppercase in 'borough_std'")
    df = fix_inconsistencies(df, log)
    df = detect_and_treat_outliers(df, count_cols, log)
    missing_pct_after = (df.isna() | (df == '')).mean().to_dict()
    log['missing_pct_after'] = {k: float(v) for k, v in missing_pct_after.items()}
    log['rows_after_clean'] = int(df.shape[0])
    out_csv = f"{out_prefix}.csv"
    out_log = f"{out_prefix}_cleaning_log.json"
    df.to_csv(out_csv, index=False)
    with open(out_log, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"Wrote cleaned CSV: {out_csv}")
    print(f"Wrote cleaning log: {out_log}")
    print("Summary of key log entries:")
    print(json.dumps({k: log[k] for k in ('start_rows','rows_after_clean','count_cols_detected') if k in log}, indent=2))

if __name__ == '__main__':
    # Chamadas diretas foram movidas para o notebook main.ipynb para execução interativa.
    pass
