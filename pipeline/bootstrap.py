"""
Bootstrap confidence-interval analysis for alignment scores.

Entry point
-----------
run_bootstrap(db_path, question_set_key, model_registry, output_dir, ...)

For each socio-demographic variable the function:
  1. Loads alignment data from the demographics table and ESS demographic columns.
  2. Derives all SD variables via recode_demographics().
  3. Runs parallelised bootstrap resampling.
  4. Computes percentile-based CIs.
  5. Writes one CSV per SD variable + alignment_overall_population.csv.

CSV schema (per SD variable)
----------------------------
variable, group, model, mean_alignment, median_alignment, std_alignment,
ci_lower, ci_upper, ci_width, n_bootstrap, n_respondents_mean
"""

import sqlite3
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    SD_VARIABLES, MODEL_REGISTRY,
    N_BOOTSTRAP, CONFIDENCE_LEVEL, RANDOM_SEED, N_PROCESSES, BATCH_SIZE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Western-country ISO-3166-1-alpha-2 set (Europe broadly + Anglosphere)
# Used to classify immigration background.
# Excludes Russia (RU) and Turkey (TR) which are typically treated as
# non-Western.
# ─────────────────────────────────────────────────────────────────────────────
_WESTERN_ISO: frozenset = frozenset({
    # EU member states
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI",
    "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK",
    # EEA / EFTA / UK
    "CH", "GB", "IS", "LI", "NO",
    # European non-EU (Balkans, Eastern Europe)
    "AL", "BA", "BY", "MD", "ME", "MK", "RS", "UA", "XK",
    # Micro-states
    "AD", "MC", "SM", "VA",
    # Anglosphere outside Europe
    "US", "CA", "AU", "NZ",
})


def _is_western_country(iso) -> "bool | None":
    """Return True=Western, False=Non-Western, None=unknown/missing."""
    if pd.isna(iso):
        return None
    code = str(iso).strip().upper()
    if code in ("6666", "7777", "8888", "9999", ""):
        return None
    return code in _WESTERN_ISO


# ─────────────────────────────────────────────────────────────────────────────
# Demographic recoding
# ─────────────────────────────────────────────────────────────────────────────

def recode_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive all SD_VARIABLES from raw ESS columns.

    Operates on a copy; original DataFrame is not modified.
    Missing source columns produce a warning and are skipped.
    """
    df = df.copy()

    # ── Generations ────────────────────────────────────────────────────────────
    if "agea" in df.columns:
        bins   = [0, 26, 43, 59, 78, 999]
        labels = ["Gen Z", "Millennials", "Gen X", "Baby Boomers", "Silent Generation"]
        df["generations"] = pd.cut(
            df["agea"], bins=bins, labels=labels, include_lowest=True
        ).astype(object)
        df.loc[df["agea"] >= 999, "generations"] = np.nan

    # ── Education (ISCED brackets) ──────────────────────────────────────────────
    if "edulvlb" in df.columns:
        def _edu(val):
            if pd.isna(val) or val in [5555, 7777, 8888, 9999]:
                return np.nan
            if val < 200: return "0-1: ≤ Lower secondary"
            if val < 300: return "2: Lower secondary"
            if val < 400: return "3: Upper secondary"
            if val < 500: return "4: Post-secondary non-tertiary"
            if val < 600: return "5: Short-cycle tertiary"
            if val < 700: return "6: Bachelor"
            if val < 800: return "7: Master"
            return "8: Doctoral"
        df["education_bracket"] = df["edulvlb"].apply(_edu)

    # ── Gender ──────────────────────────────────────────────────────────────────
    if "nobingnd" in df.columns:
        df["gender_clean"] = df["nobingnd"].apply(
            lambda x: x if pd.notna(x) and x in [1, 2, 3] else np.nan
        )

    # ── Ethnic majority ─────────────────────────────────────────────────────────
    if "feethngr" in df.columns:
        df["ethnicmajority_clean"] = df["feethngr"].apply(
            lambda x: x if pd.notna(x) and x in [1, 2] else np.nan
        )

    # ── Income decile ───────────────────────────────────────────────────────────
    if "hinctnta" in df.columns:
        df["income_clean"] = df["hinctnta"].apply(
            lambda x: float(x) if pd.notna(x) and 1 <= x <= 10 else np.nan
        )

    # ── Religion (belonging + denomination) ────────────────────────────────────
    # rlgblg == 2 → 0.0 (No Religion); rlgblg == 1 → denomination from rlgdnm.
    _denom_col = next((c for c in ["rlgdnm", "rlgdnm_clean"] if c in df.columns), None)
    if "rlgblg" in df.columns:
        denom = (
            df[_denom_col].apply(
                lambda x: float(x) if pd.notna(x) and x not in [55, 66, 77, 88, 99]
                else np.nan
            )
            if _denom_col
            else pd.Series(np.nan, index=df.index)
        )
        blg = df["rlgblg"]
        blg_valid = blg.isin([1, 2])
        religion = pd.Series(np.nan, index=df.index)
        religion[blg_valid & (blg == 2)] = 0.0
        religion[blg_valid & (blg == 1)] = denom[blg_valid & (blg == 1)]
        df["religion_clean"] = religion
    elif _denom_col:
        df["religion_clean"] = df[_denom_col].apply(
            lambda x: float(x) if pd.notna(x) and x not in [55, 66, 77, 88, 99]
            else np.nan
        )

    # ── Religiosity bracket ─────────────────────────────────────────────────────
    if "rlgdgr" in df.columns:
        def _relig(val):
            if pd.isna(val) or val in [77, 88, 99] or not (0 <= val <= 10):
                return np.nan
            if val <= 2: return "0-2: Low religiosity"
            if val <= 7: return "3-7: Medium religiosity"
            return "8-10: High religiosity"
        df["religiosity_clean"] = df["rlgdgr"].apply(_relig)

    # ── Political interest ──────────────────────────────────────────────────────
    if "polintr" in df.columns:
        df["political_interest_clean"] = df["polintr"].apply(
            lambda x: float(x) if pd.notna(x) and x in [1, 2, 3, 4] else np.nan
        )

    # ── Political involvement (sum of 7 binary participation items) ─────────────
    _pol_vars = ["contplt", "donprty", "badge", "sgnptit", "pbldmna", "bctprd", "pstplonl"]
    if all(v in df.columns for v in _pol_vars):
        def _pol_inv(row):
            score, valid = 0, 0
            for v in _pol_vars:
                val = row[v]
                if pd.notna(val) and val in [1, 2]:
                    valid += 1
                    if val == 1:
                        score += 1
            return float(score) if valid >= 4 else np.nan
        df["political_involvement"] = df.apply(_pol_inv, axis=1)

    # ── Internet time bracket ───────────────────────────────────────────────────
    if "netustm" in df.columns:
        def _itime(val):
            if pd.isna(val) or val in [6666, 7777, 8888, 9999] or val < 0:
                return np.nan
            if val == 0:   return "0: No use"
            if val <= 30:  return "1-30: Up to 1/2 hour"
            if val <= 60:  return "31-60: 1/2-1 hour"
            if val <= 180: return "61-180: 1-3 hours"
            if val <= 300: return "181-300: 3-5 hours"
            if val <= 420: return "301-420: 5-7 hours"
            return "421+: More than 7 hours"
        df["internet_time_bracket"] = df["netustm"].apply(_itime)

    # ── Domicile ────────────────────────────────────────────────────────────────
    # Already pre-computed in demographics as domicil_clean (numeric 1-5).
    # If not present, derive from domicil.
    if "domicil_clean" not in df.columns and "domicil" in df.columns:
        df["domicil_clean"] = df["domicil"].apply(
            lambda x: float(x) if pd.notna(x) and 1 <= x <= 5 else np.nan
        )

    # ── Main activity ────────────────────────────────────────────────────────────
    for col in ["mnactic", "mainact"]:
        if col in df.columns:
            df["mnactic_clean"] = df[col].apply(
                lambda x: float(x) if pd.notna(x) and 1 <= x <= 9 else np.nan
            )
            break

    # ── ISCO-08 major group ─────────────────────────────────────────────────────
    # Major group = first digit of 4-digit ISCO code.
    for col in ["iscoco", "isco08c", "isco08", "occb"]:
        if col in df.columns:
            def _isco(val):
                if pd.isna(val) or val in [9997, 9998, 9999]:
                    return np.nan
                major = int(val) // 1000
                return float(major) if 0 <= major <= 9 else np.nan
            df["isco08_major_group"] = df[col].apply(_isco)
            break

    # ── Immigration background ──────────────────────────────────────────────────
    # Categories: Native / Non-Native Western / Non-Native Non-Western /
    #             Non-Native >=1 Western Parent / Non-Native Both Parents Non-Western
    _imm_required = ["brncntr", "facntr", "mocntr"]
    if all(c in df.columns for c in _imm_required):
        _has_cntbrthd = "cntbrthd" in df.columns
        _has_fbrncntc = "fbrncntc" in df.columns
        _has_mbrncntc = "mbrncntc" in df.columns

        def _immbg(row):
            brn = row["brncntr"]
            if pd.isna(brn) or brn in [7, 8, 9]:
                return np.nan

            fac = row["facntr"]
            moc = row["mocntr"]
            fac_valid = pd.notna(fac) and fac in [1, 2]
            moc_valid = pd.notna(moc) and moc in [1, 2]

            # No migration background: born in survey country + both parents too
            if brn == 1:
                if (not fac_valid or fac == 1) and (not moc_valid or moc == 1):
                    return "No Migration Background"

            # Classify by parents' birth countries (primary signal).
            # Only fall back to respondent's own birth country when born abroad
            # AND no parent-country info is available at all.
            origins: list[bool] = []

            if fac_valid and fac == 2 and _has_fbrncntc:
                w = _is_western_country(row["fbrncntc"])
                if w is not None:
                    origins.append(w)
            if moc_valid and moc == 2 and _has_mbrncntc:
                w = _is_western_country(row["mbrncntc"])
                if w is not None:
                    origins.append(w)

            # Fallback: respondent born abroad, no parent-country data available
            if brn == 2 and not origins and _has_cntbrthd:
                w = _is_western_country(row["cntbrthd"])
                if w is not None:
                    origins.append(w)

            if not origins:
                return np.nan
            if all(origins):
                return "Western Migration Background"
            if not any(origins):
                return "Non-Western Migration Background"
            return "Mixed Migration Background"

        df["immigration_background"] = df.apply(_immbg, axis=1)

    # ── Household income feeling ────────────────────────────────────────────────
    if "hincfel" in df.columns:
        df["hincfel_clean"] = df["hincfel"].apply(
            lambda x: float(x) if pd.notna(x) and x in [1, 2, 3, 4] else np.nan
        )

    # ── Childhood financial difficulties ───────────────────────────────────────
    if "fnsdfml" in df.columns:
        df["fnsdfml_clean"] = df["fnsdfml"].apply(
            lambda x: float(x) if pd.notna(x) and x in [1, 2, 3, 4, 5] else np.nan
        )

    # ── Country ──────────────────────────────────────────────────────────────────
    # cntry is already present from ess11; no transformation needed.

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap worker (runs in child process)
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_worker_all_vars(args: tuple) -> list[tuple]:
    """
    Process a batch of bootstrap samples for ALL SD variables simultaneously.

    For each bootstrap sample the alignment sub-matrix is computed once and
    reused across every variable grouping, avoiding redundant indexing.

    Parameters passed via a single tuple to be compatible with Pool.map:
      aligns_matrix     (N, M) float64 array  – NaN where alignment is missing
      all_group_codes   list of V (N,) int32 arrays  – -1 = missing group
      all_unique_groups list of V lists of group values
      model_shorts      list of M short model names
      boot_batches      list of (boot_idx, sample_indices_array) tuples

    Returns a flat list of tuples:
      (v_idx, boot_idx, group_val, model_short, mean_alignment, n_respondents)
    """
    aligns_matrix, all_group_codes, all_unique_groups, model_shorts, boot_batches = args
    results = []

    for boot_idx, sample_indices in boot_batches:
        sa = aligns_matrix[sample_indices]   # (N, M) – computed once per sample

        for v_idx, (group_codes, unique_groups) in enumerate(
            zip(all_group_codes, all_unique_groups)
        ):
            sg = group_codes[sample_indices]     # (N,)

            for g_code, g_val in enumerate(unique_groups):
                g_mask = sg == g_code
                if g_mask.sum() == 0:
                    continue
                g_aligns = sa[g_mask]            # (n_g, M)

                for m_idx, ms in enumerate(model_shorts):
                    vals = g_aligns[:, m_idx]
                    valid = vals[~np.isnan(vals)]
                    if len(valid) == 0:
                        continue
                    results.append((
                        v_idx, boot_idx, g_val, ms,
                        float(np.mean(valid)), int(len(valid)),
                    ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main bootstrap function
# ─────────────────────────────────────────────────────────────────────────────

def run_bootstrap(
    db_path,
    question_set_key: str,
    model_registry: dict = MODEL_REGISTRY,
    sd_variables: list = SD_VARIABLES,
    output_dir: Path = None,
    n_bootstrap: int = N_BOOTSTRAP,
    conf_level: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
    n_processes: int = N_PROCESSES,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Run bootstrap analysis and write CSV results.

    Reads alignment columns from the demographics table
    ({model_short}_{question_set_key}_alignment) then joins with ess11 to
    derive socio-demographic variables.

    Output CSVs written to output_dir:
      alignment_{var}.csv           – per-variable bootstrap summary
      alignment_overall_population.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Identify available model alignment columns ──────────────────────────
    active_model_shorts = []
    align_cols_db = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(demographics)")
    demo_cols = {r[1] for r in cursor.fetchall()}
    conn.close()

    for ms in model_registry.values():
        col = f"{ms}_{question_set_key}_alignment"
        if col in demo_cols:
            active_model_shorts.append(ms)
            align_cols_db.append(col)

    if not active_model_shorts:
        raise RuntimeError(
            f"No alignment columns found for question_set '{question_set_key}' "
            "in demographics table. Run the alignment step first."
        )
    print(f"  Bootstrap: {len(active_model_shorts)} models × {n_bootstrap} samples")

    # ── Load data ────────────────────────────────────────────────────────────
    col_select = ", ".join(f'd."{c}"' for c in align_cols_db)
    ess_cols = (
        "e.new_id, e.cntry, e.agea, e.edulvlb, e.nobingnd, e.feethngr, "
        "e.hinctnta, e.rlgdgr, e.netustm, e.domicil, e.polintr, "
        "e.contplt, e.donprty, e.badge, e.sgnptit, e.pbldmna, e.bctprd, e.pstplonl"
    )
    # Optional columns (may not exist) — include all known ISCO column names
    optional = [
        "e.rlgdnm", "e.mnactic", "e.iscoco", "e.isco08", "e.isco08c", "e.occb",
        "e.rlgblg", "e.brncntr", "e.cntbrthd", "e.facntr", "e.fbrncntc",
        "e.mocntr", "e.mbrncntc", "e.hincfel", "e.fnsdfml",
    ]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(ess11)")
    ess_col_names = {r[1] for r in cursor.fetchall()}
    cursor.execute("PRAGMA table_info(demographics)")
    demo_col_names = {r[1] for r in cursor.fetchall()}
    conn.close()

    opt_available = [c.split(".")[1] for c in optional
                     if c.split(".")[1] in ess_col_names]
    if opt_available:
        ess_cols += ", " + ", ".join(f"e.{c}" for c in opt_available)
    # Also pull domicil_clean from demographics if already computed
    if "domicil_clean" in demo_col_names:
        col_select += ', d."domicil_clean"'
    if "rlgdnm_clean" in demo_col_names:
        col_select += ', d."rlgdnm_clean"'

    query = f"""
        SELECT {col_select}, {ess_cols}
        FROM demographics d
        JOIN ess11 e ON d.new_id = e.new_id
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Rename alignment columns to generic short form for easier access
    rename_map = {c: ms for ms, c in zip(active_model_shorts, align_cols_db)}
    df = df.rename(columns=rename_map)

    # ── Derive demographic variables ─────────────────────────────────────────
    df = recode_demographics(df)

    # ── Pre-generate bootstrap sample indices ────────────────────────────────
    rng = np.random.default_rng(seed)
    N = len(df)
    boot_indices = [
        rng.integers(0, N, size=N) for _ in range(n_bootstrap)
    ]

    # ── Build alignment matrix ───────────────────────────────────────────────
    align_matrix = df[active_model_shorts].to_numpy(dtype=float)  # (N, M)

    # ── Determine actual n_processes ─────────────────────────────────────────
    if n_processes is None:
        n_processes = max(1, int(cpu_count() * 0.8))

    # ── Pre-compute per-iteration overall means: (n_bootstrap, n_models) ─────
    # overall_boot_means[b, m] = mean alignment of model m in bootstrap iteration b
    # Used to compute cross-model deviation CIs in each variable bootstrap.
    overall_boot_means = np.array([
        np.nanmean(align_matrix[idx], axis=0)
        for idx in boot_indices
    ])  # shape (n_bootstrap, n_models)

    # ── Overall population alignment (no sub-grouping) ───────────────────────
    _write_overall_population(
        df, active_model_shorts, align_matrix,
        boot_indices, conf_level, output_dir,
    )

    # ── Encode all SD variables upfront ──────────────────────────────────────
    missing_vars = [v for v in sd_variables if v not in df.columns]
    if missing_vars:
        warnings.warn(f"SD variables not available (skipped): {missing_vars}")

    all_group_codes: list   = []
    all_unique_groups: list = []
    available_vars: list    = []
    for var in sd_variables:
        if var not in df.columns:
            continue
        series = df[var]
        unique_groups = sorted(series.dropna().unique())
        if len(unique_groups) == 0:
            warnings.warn(f"  Skipping {var}: no non-null values")
            continue
        valid_mask    = series.notna()
        group_to_code = {g: i for i, g in enumerate(unique_groups)}
        group_codes   = np.where(
            valid_mask,
            series.map(group_to_code).fillna(-1).astype(int),
            -1,
        )
        all_group_codes.append(group_codes)
        all_unique_groups.append(unique_groups)
        available_vars.append(var)

    # ── Single combined bootstrap pass across all SD variables ────────────────
    print(f"  Bootstrap: {n_bootstrap} samples × {len(available_vars)} SD variables")
    indexed = list(enumerate(boot_indices))
    batches = [
        indexed[i:i + batch_size] for i in range(0, len(indexed), batch_size)
    ]
    worker_args = [
        (align_matrix, all_group_codes, all_unique_groups, active_model_shorts, batch)
        for batch in batches
    ]

    if n_processes > 1:
        with Pool(processes=n_processes) as pool:
            all_raw = pool.map(_bootstrap_worker_all_vars, worker_args)
    else:
        all_raw = [_bootstrap_worker_all_vars(a) for a in worker_args]

    flat = [row for batch_res in all_raw for row in batch_res]
    df_boot_all = pd.DataFrame(
        flat, columns=["v_idx", "boot_idx", "group", "model", "mean", "n"]
    )

    # ── Write per-variable summaries (aggregation only) ───────────────────────
    for i, var in enumerate(available_vars):
        print(f"  Summarising → {var}")
        df_boot_var = df_boot_all[df_boot_all["v_idx"] == i].drop(columns="v_idx")
        _summarize_and_write_variable(
            df_boot_var, var, all_unique_groups[i], active_model_shorts,
            conf_level, output_dir,
            overall_boot_means=overall_boot_means,
        )


def _compute_cross_model_deviation(
    df_boot: pd.DataFrame,
    overall_boot_means: np.ndarray,   # shape (n_bootstrap, n_models)
    model_shorts: list,
    unique_groups: list,
    conf_level: float,
) -> pd.DataFrame:
    """
    For each group, compute the bootstrap distribution of the cross-model
    mean deviation:
        cross_dev[b, g] = mean_m ( group_mean[b,g,m] − overall_mean[b,m] )

    CI is the percentile interval of that distribution over b.

    Returns a DataFrame with columns:
        group, mean_dev, ci_lower_dev, ci_upper_dev, ci_width_dev,
        n_bootstrap, n_models, n_respondents_mean
    """
    alpha = 1 - conf_level

    # Build a (n_bootstrap, n_models) DataFrame of overall means
    overall_df = pd.DataFrame(overall_boot_means, columns=model_shorts)
    overall_df.index.name = "boot_idx"
    overall_df = overall_df.reset_index()
    overall_melt = overall_df.melt(
        id_vars="boot_idx", var_name="model", value_name="overall_mean"
    )

    # Merge per-bootstrap group means with overall means on (boot_idx, model)
    merged = df_boot.merge(overall_melt, on=["boot_idx", "model"], how="inner")
    merged["dev"] = merged["mean"] - merged["overall_mean"]

    # Average deviation across models for each (boot_idx, group)
    cross = (
        merged.groupby(["boot_idx", "group"])["dev"]
        .mean()
        .reset_index()
        .rename(columns={"dev": "cross_dev"})
    )

    # Average respondent count per group
    n_resp = (
        df_boot.groupby(["group", "model"])["n"].mean()
        .groupby("group").mean()
    )

    records = []
    for g in unique_groups:
        g_devs = cross[cross["group"] == g]["cross_dev"].dropna().values
        if len(g_devs) == 0:
            continue
        n_mod = float(merged[merged["group"] == g]["model"].nunique())
        nr = float(n_resp.get(g, np.nan)) if g in n_resp.index else np.nan
        records.append({
            "group":              g,
            "mean_dev":           float(np.mean(g_devs)),
            "ci_lower_dev":       float(np.percentile(g_devs, 100 * alpha / 2)),
            "ci_upper_dev":       float(np.percentile(g_devs, 100 * (1 - alpha / 2))),
            "ci_width_dev":       float(
                np.percentile(g_devs, 100 * (1 - alpha / 2))
                - np.percentile(g_devs, 100 * alpha / 2)
            ),
            "n_bootstrap":        len(g_devs),
            "n_models":           n_mod,
            "n_respondents_mean": nr,
        })
    return pd.DataFrame(records)


def _write_overall_population(df, model_shorts, align_matrix,
                               boot_indices, conf_level, output_dir):
    """Write overall population bootstrap summary (no demographic split)."""
    records = []
    for m_idx, ms in enumerate(model_shorts):
        vals = align_matrix[:, m_idx]
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            continue
        boot_means = np.array([
            np.nanmean(vals[idx]) for idx in boot_indices
        ])
        alpha = 1 - conf_level
        records.append({
            "variable":         "overall",
            "group":            "All respondents",
            "model":            ms,
            "mean_alignment":   float(np.mean(boot_means)),
            "median_alignment": float(np.median(boot_means)),
            "std_alignment":    float(np.std(boot_means, ddof=1)),
            "ci_lower":         float(np.percentile(boot_means, 100 * alpha / 2)),
            "ci_upper":         float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
            "ci_width":         float(np.percentile(boot_means, 100 * (1 - alpha / 2))
                                      - np.percentile(boot_means, 100 * alpha / 2)),
            "n_bootstrap":      len(boot_means),
            "n_respondents_mean": float(len(valid)),
        })

    pd.DataFrame(records).to_csv(
        output_dir / "alignment_overall_population.csv", index=False
    )
    print(f"  Written: alignment_overall_population.csv")


def _summarize_and_write_variable(
    df_boot: pd.DataFrame,
    var: str,
    unique_groups: list,
    model_shorts: list,
    conf_level: float,
    output_dir: Path,
    overall_boot_means=None,
) -> None:
    """Aggregate pre-computed bootstrap rows for one SD variable and write CSVs."""
    if df_boot.empty:
        warnings.warn(f"  No bootstrap results for {var}")
        return

    alpha = 1 - conf_level
    summary = []
    for group in unique_groups:
        for ms in model_shorts:
            sub = df_boot[(df_boot["group"] == group) & (df_boot["model"] == ms)]
            if sub.empty:
                continue
            vals = sub["mean"].values
            summary.append({
                "variable":           var,
                "group":              group,
                "model":              ms,
                "mean_alignment":     float(np.mean(vals)),
                "median_alignment":   float(np.median(vals)),
                "std_alignment":      float(np.std(vals, ddof=1)),
                "ci_lower":           float(np.percentile(vals, 100 * alpha / 2)),
                "ci_upper":           float(np.percentile(vals, 100 * (1 - alpha / 2))),
                "ci_width":           float(
                    np.percentile(vals, 100 * (1 - alpha / 2))
                    - np.percentile(vals, 100 * alpha / 2)
                ),
                "n_bootstrap":        len(vals),
                "n_respondents_mean": float(sub["n"].mean()),
            })

    pd.DataFrame(summary).to_csv(output_dir / f"alignment_{var}.csv", index=False)
    print(f"  Written: alignment_{var}.csv  ({len(summary)} rows)")

    if overall_boot_means is not None:
        dev_df = _compute_cross_model_deviation(
            df_boot, overall_boot_means, model_shorts, unique_groups, conf_level
        )
        if not dev_df.empty:
            dev_df.insert(0, "variable", var)
            dev_path = output_dir / f"alignment_deviation_{var}.csv"
            dev_df.to_csv(dev_path, index=False)
            print(f"  Written: alignment_deviation_{var}.csv  ({len(dev_df)} rows)")
