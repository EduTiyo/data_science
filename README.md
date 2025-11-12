This README documents the cleaning pipeline implemented in `main.ipynb` (download + chunked cleaning) and the decisions applied during processing.

Summary of transformations and decisions

1) Missing values
- Count-columns (detected by `data_cleaning.is_count_col` e.g. `number_of_persons_injured`) are coerced to numeric and imputed with 0. They are stored using pandas nullable integer dtype `Int64` to preserve the ability to represent NA if needed. Rationale: in most records missing indicates 0 events; however we also create indicators to preserve that the value was originally missing (columns like `{col}_was_missing` are available in the cleaned output).
- Date/time fields that cannot be parsed are left as NaT (no automatic imputation).

2) Outliers
- For count columns we apply per-chunk winsorization by default using the 1% / 99% quantiles as caps. For each column we create `{col}_is_outlier` (boolean) and `{col}_outlier_orig` containing the original value for auditoria. The per-chunk thresholds and number of outliers are recorded in the chunk-level `outlier_log`.

3) Inconsistencies (dependencies between variables)
- The pipeline detects when the sum of component columns (e.g., all `_injured` columns) exceeds the reported total (`number_of_persons_injured` / `number_of_persons_killed`). It calls `data_cleaning.fix_inconsistencies` to apply domain rules and records counts before/after fix in the per-chunk log. Simple policy used in the pipeline: treat components as authoritative and update the total to the summed components (the action and counts are logged).

4) Standardization (formats)
- `crash_date` -> `crash_date_parsed`: parsed via `pd.to_datetime` (unparsable -> NaT). Prefer storing as `datetime64[ns]` for analysis; current pipeline stores `.dt.date` but keeps parseability in logs.
- `crash_time` -> `crash_time_parsed`: parsed trying `%H:%M`, `%H:%M:%S`, then a generic parse; unparsable -> NaT. Consider building a combined `crash_datetime` if both date and time exist.
- `zip_code` -> `zip_code_std`: normalized via `data_cleaning.standardize_zip` (keeps 5-digit strings, left-pads zeros when needed).
- `borough` -> `borough_std`: trimmed and uppercased; `'NONE'` and empty strings set to None.
- `latitude` / `longitude`: coerced to float; zeros converted to NaN (observed as invalid placeholders). No imputation is applied.

5) Audit columns and logs
- For transparency the pipeline creates several audit/flag columns per row where applicable, for example:
	- `{count_col}_was_missing` — True if the count column was missing before imputation
	- `{count_col}_is_outlier`, `{count_col}_outlier_orig` — outlier flags and original values
	- `injured_inconsistency`, `injured_correction` — flags/actions for inconsistency fixes

- Aggregated logs are written to `cleaned_logs.json` (per the notebook): it contains per-chunk summaries (rows processed, counts missing before/after, outlier summaries, inconsistencies before/after) and a `decisions` section describing the cleaning policies applied.

6) Files produced by the notebook
- `data.csv` — RAW paginated download (append-only) produced by the download cell.
- `logs.json` — download log with chunks and rows downloaded.
- `cleaned_data.csv` — cleaned dataset produced by the cleaning cell (appended per chunk).
- `cleaned_logs.json` — aggregated cleaning log documenting per-chunk stats and the `decisions` map (policies applied).

7) Next steps / recommendations
- Re-evaluate the choice to impute 0 for counts with stakeholders; if "missing" means "unknown" rather than 0, consider keeping NA or using model-based imputation and keep an indicator column.
- Consider computing global thresholds for outlier detection (calibrated on a sample) instead of per-chunk quantiles to avoid inconsistent capping across chunks.
- Add deduplication (check `collision_id` or other unique ids) and referential checks where appropriate.
- Add unit tests for invariants (e.g., no negative counts, component sums ≤ total after fix, datetime parse rates) and a lightweight validation run (`--limit 1000`) to produce a small log for review.

If you want, I can now:
- run a short validation (e.g., set `target_rows = 1000`) and paste the resulting `cleaned_logs.json` summary here, or
- embed this documentation into a new notebook cell (Markdown) so it travels with the pipeline.