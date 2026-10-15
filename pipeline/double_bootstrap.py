"""
Double-bootstrap alignment uncertainty analysis.

Two sources of uncertainty are jointly captured per iteration:

  1. **Model-answer bootstrap**: for each (model, question) pair, draw
     ``pool_size`` (= 20) answers with replacement from the observed
     answer pool stored in the ``all_answers`` column of the majority-votes
     CSV (NaN entries represent refusals).  Compute the majority vote from
     the bootstrapped draw.  A question is *active* for this iteration only
     when **all** models produce a valid (non-NaN) majority vote.

  2. **Population bootstrap**: resample ESS respondents with replacement,
     exactly as in ``run_bootstrap``.

Output CSV schema is identical to ``run_bootstrap`` so all existing plot
functions work without modification.  Written to ``output_dir/``:

    alignment_overall_population.csv
    alignment_<var>.csv              – per SD variable
    alignment_deviation_<var>.csv    – cross-model deviation CIs
"""

from __future__ import annotations

import ast
import re
import sqlite3
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BATCH_SIZE,
    CONFIDENCE_LEVEL,
    MODEL_REGISTRY,
    N_BOOTSTRAP,
    QUESTION_GROUP_MAP,
    RANDOM_SEED,
    SD_VARIABLES,
)
from .bootstrap import (
    _compute_cross_model_deviation,
    _summarize_and_write_variable,
    recode_demographics,
)

# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_answers(s, expected_size: int = 20) -> np.ndarray:
    """Parse the ``all_answers`` column (Python list repr) → float64 array.

    Handles bare ``nan`` tokens produced by pandas' ``str(list)``
    representation.  Pads with NaN or truncates to *expected_size*.
    """
    if pd.isna(s):
        return np.full(expected_size, np.nan, dtype=np.float64)
    cleaned = re.sub(r"\bnan\b", "None", str(s))
    try:
        lst = ast.literal_eval(cleaned)
        arr = np.array(
            [np.nan if v is None else float(v) for v in lst],
            dtype=np.float64,
        )
        if len(arr) < expected_size:
            arr = np.concatenate(
                [arr, np.full(expected_size - len(arr), np.nan, dtype=np.float64)]
            )
        return arr[:expected_size]
    except (ValueError, SyntaxError):
        return np.full(expected_size, np.nan, dtype=np.float64)


def _majority_vote_1d(arr: np.ndarray) -> float:
    """Mode of non-NaN entries; NaN if all NaN.  Ties resolved by minimum."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.nan
    unique, counts = np.unique(valid, return_counts=True)
    return float(unique[counts == counts.max()].min())


# ─────────────────────────────────────────────────────────────────────────────
# Global worker state
# Populated by run_double_bootstrap() *before* the Pool is created.
# Child processes inherit it via fork (Linux default) without any pickling
# overhead — the large numpy arrays are shared read-only in the OS page-cache.
# ─────────────────────────────────────────────────────────────────────────────

_G: dict = {}


def _dbl_boot_worker(batch: list) -> dict:
    """Process one batch of double-bootstrap iterations.

    Each element of *batch* is ``(boot_idx, model_seed, pop_seed)``.

    Returns
    -------
    dict with keys:
      ``"groups"``  : list of (v_idx, boot_idx, group, model, mean, n)
      ``"overall"`` : list of (boot_idx, model, mean, n)
    """
    answer_pool     = _G["answer_pool"]      # (M, Q, pool_size) float64
    ans_matrix      = _G["ans_matrix"]       # (N, Q) float64
    resp_valid_mask = _G["resp_valid_mask"]  # (N, Q) bool
    scale_ranges    = _G["scale_ranges"]     # (Q,) float64
    indicator_full  = _G["indicator_full"]   # (Q, G) float64
    all_group_codes = _G["all_group_codes"]  # list of (N,) int32
    all_unique_grps = _G["all_unique_grps"]  # list of sorted lists
    model_shorts    = _G["model_shorts"]     # list[str]

    N_models, N_questions, pool_size = answer_pool.shape
    N_respondents = ans_matrix.shape[0]

    group_results:   list = []
    overall_results: list = []

    for boot_idx, model_seed, pop_seed in batch:
        rng_m = np.random.default_rng(model_seed)
        rng_p = np.random.default_rng(pop_seed)

        # ── 1. Bootstrap model answers → majority votes ───────────────────
        # draw_idx[mi, qi, :] = pool_size indices drawn w.r. for model mi, q qi
        draw_idx = rng_m.integers(
            0, pool_size, size=(N_models, N_questions, pool_size)
        )
        sampled = answer_pool[
            np.arange(N_models)[:, None, None],
            np.arange(N_questions)[None, :, None],
            draw_idx,
        ]  # (M, Q, pool_size)

        votes = np.empty((N_models, N_questions), dtype=np.float64)
        for mi in range(N_models):
            for qi in range(N_questions):
                votes[mi, qi] = _majority_vote_1d(sampled[mi, qi])

        # ── 2. Active question set ────────────────────────────────────────
        # A question is active only when ALL models produce a valid vote.
        active_q = ~np.any(np.isnan(votes), axis=0)  # (Q,) bool
        if not active_q.any():
            continue  # degenerate iteration — skip

        # Restrict the group-indicator matrix to active questions: (Q, G)
        indicator = indicator_full * active_q[:, None]

        # ── 3. Per-model person-level alignment ───────────────────────────
        align_matrix = np.full((N_respondents, N_models), np.nan, dtype=np.float64)

        for mi in range(N_models):
            votes_m = votes[mi]  # (Q,)
            # NaN where respondent has no valid answer or question is inactive
            valid_mask = resp_valid_mask & active_q[None, :]  # (N, Q)
            with np.errstate(invalid="ignore"):
                pq_align = np.where(
                    valid_mask,
                    1.0 - np.abs(ans_matrix - votes_m[None, :]) / scale_ranges[None, :],
                    np.nan,
                )  # (N, Q)

            # Question-group averaging via matrix multiply
            pq_fill  = np.where(np.isnan(pq_align), 0.0, pq_align)   # (N, Q)
            pq_count = (~np.isnan(pq_align)).astype(np.float64)       # (N, Q)

            grp_sum   = pq_fill  @ indicator   # (N, G)
            grp_count = pq_count @ indicator   # (N, G)
            # np.where evaluates both branches, so the zero-count rows still
            # divide 0/0 before being discarded.
            with np.errstate(invalid="ignore"):
                grp_means = np.where(grp_count > 0.0, grp_sum / grp_count, np.nan)

            # Person-level score = nanmean across question-groups
            grp_f   = np.where(np.isnan(grp_means), 0.0, grp_means)
            grp_c   = (~np.isnan(grp_means)).astype(np.float64)
            total_c = grp_c.sum(axis=1)  # (N,)
            with np.errstate(invalid="ignore"):
                person_align = np.where(
                    total_c > 0.0,
                    grp_f.sum(axis=1) / total_c,
                    np.nan,
                )
            align_matrix[:, mi] = person_align

        # ── 4. Population bootstrap ───────────────────────────────────────
        pop_idx = rng_p.integers(0, N_respondents, size=N_respondents)
        sa      = align_matrix[pop_idx]  # (N, M)

        # Overall mean per model over the bootstrapped population sample
        for m_idx, ms in enumerate(model_shorts):
            vals  = sa[:, m_idx]
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                continue
            overall_results.append(
                (boot_idx, ms, float(np.mean(valid)), int(len(valid)))
            )

        # ── 5. SD-variable group means ────────────────────────────────────
        for v_idx, (g_codes, u_groups) in enumerate(
            zip(all_group_codes, all_unique_grps)
        ):
            sg = g_codes[pop_idx]  # (N,)
            for g_code, g_val in enumerate(u_groups):
                g_mask  = sg == g_code
                if not g_mask.any():
                    continue
                g_aligns = sa[g_mask]  # (n_g, M)
                for m_idx, ms in enumerate(model_shorts):
                    vals  = g_aligns[:, m_idx]
                    valid = vals[~np.isnan(vals)]
                    if len(valid) == 0:
                        continue
                    group_results.append(
                        (
                            v_idx, boot_idx, g_val, ms,
                            float(np.mean(valid)), int(len(valid)),
                        )
                    )

    return {"groups": group_results, "overall": overall_results}


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_overall(
    df_overall_boot: pd.DataFrame,
    model_shorts: list,
    conf_level: float,
    output_dir: Path,
) -> None:
    """Aggregate per-iteration overall means → ``alignment_overall_population.csv``."""
    alpha   = 1.0 - conf_level
    records = []
    for ms in model_shorts:
        sub  = df_overall_boot[df_overall_boot["model"] == ms]
        vals = sub["mean"].values
        if len(vals) == 0:
            continue
        lo = float(np.percentile(vals, 100.0 * alpha / 2.0))
        hi = float(np.percentile(vals, 100.0 * (1.0 - alpha / 2.0)))
        records.append(
            {
                "variable":           "overall",
                "group":              "All respondents",
                "model":              ms,
                "mean_alignment":     float(np.mean(vals)),
                "median_alignment":   float(np.median(vals)),
                "std_alignment":      float(np.std(vals, ddof=1)),
                "ci_lower":           lo,
                "ci_upper":           hi,
                "ci_width":           hi - lo,
                "n_bootstrap":        int(len(vals)),
                "n_respondents_mean": float(sub["n"].mean()),
            }
        )
    pd.DataFrame(records).to_csv(
        output_dir / "alignment_overall_population.csv", index=False
    )
    print("  Written: alignment_overall_population.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _print_pool_diagnostics(
    answer_pool: np.ndarray,
    model_shorts: list,
    n_test: int = 200,
    seed: int = 0,
) -> None:
    """Print answer-pool statistics and vote-stability metrics.

    Helps verify that (a) parsing succeeded and (b) the model-answer layer
    is actually introducing variability into the bootstrap.
    """
    M, Q, pool_size = answer_pool.shape
    rng = np.random.default_rng(seed)

    print("\n  ── Answer pool ──────────────────────────────────────────")

    # Compute original majority votes from the full pool (no resampling)
    orig_votes = np.empty((M, Q), dtype=np.float64)
    for mi in range(M):
        for qi in range(Q):
            orig_votes[mi, qi] = _majority_vote_1d(answer_pool[mi, qi])

    orig_active = (~np.any(np.isnan(orig_votes), axis=0)).sum()
    print(f"    Questions with valid vote for ALL models: {orig_active}/{Q}")

    # Per-model pool fill rate
    for mi, ms in enumerate(model_shorts):
        pool_m        = answer_pool[mi]                            # (Q, pool_size)
        n_valid_per_q = (~np.isnan(pool_m)).sum(axis=1)           # (Q,)
        full_q        = (n_valid_per_q == pool_size).sum()
        partial_q     = ((n_valid_per_q > 0) & (n_valid_per_q < pool_size)).sum()
        all_nan_q     = (n_valid_per_q == 0).sum()
        mean_valid    = n_valid_per_q.mean()
        print(
            f"    {ms:20s}: fully-valid={full_q:3d}  partial={partial_q:3d}"
            f"  all-NaN={all_nan_q:3d}  mean valid/q={mean_valid:.1f}/{pool_size}"
        )

    # Vote-stability test: how often does resampling change the majority vote?
    print(
        f"\n  ── Vote stability ({n_test} test resamples) ────────────────"
    )
    changed_ever  = np.zeros((M, Q), dtype=bool)
    active_counts = []

    for _ in range(n_test):
        draw_idx = rng.integers(0, pool_size, size=(M, Q, pool_size))
        sampled  = answer_pool[
            np.arange(M)[:, None, None],
            np.arange(Q)[None, :, None],
            draw_idx,
        ]
        votes = np.empty((M, Q), dtype=np.float64)
        for mi in range(M):
            for qi in range(Q):
                votes[mi, qi] = _majority_vote_1d(sampled[mi, qi])

        # A change counts when both orig and new vote are non-NaN but differ
        differ        = (votes != orig_votes) & ~np.isnan(orig_votes) & ~np.isnan(votes)
        changed_ever |= differ
        active_counts.append((~np.any(np.isnan(votes), axis=0)).sum())

    pct_stable = 100.0 * (~changed_ever).sum() / (M * Q)
    print(
        f"    (model,question) pairs that NEVER changed vote: "
        f"{(~changed_ever).sum()}/{M*Q}  ({pct_stable:.1f}% stable)"
    )
    print(
        f"    Active questions per iteration: "
        f"mean={np.mean(active_counts):.1f}  "
        f"min={min(active_counts)}  max={max(active_counts)}"
    )
    for mi, ms in enumerate(model_shorts):
        n_ever_changed = changed_ever[mi].sum()
        print(f"      {ms:20s}: {n_ever_changed:3d}/{Q} questions changed vote at least once")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Per-model deviation helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_per_model_deviations(
    df_boot_all: pd.DataFrame,
    df_overall_boot: pd.DataFrame,
    available_vars: list,
    all_unique_grps: list,
    active_model_shorts: list,
    conf_level: float,
    deviations_dir: Path,
) -> None:
    """Write per-model deviation CSVs to *deviations_dir*.

    For each (variable, group, model) triple computes the bootstrap
    distribution of ``dev[b,g,m] = group_mean[b,g,m] − overall_mean[b,m]``
    and the corresponding percentile CI.

    Output: ``deviations_dir/deviation_{var}.csv`` with columns:
        variable, group, model, mean_dev, ci_lower_dev, ci_upper_dev,
        ci_width_dev, n_bootstrap, n_respondents_mean
    """
    alpha = 1.0 - conf_level
    overall_melt = df_overall_boot[["boot_idx", "model", "mean"]].rename(
        columns={"mean": "overall_mean"}
    )

    for i, var in enumerate(available_vars):
        df_var = df_boot_all[df_boot_all["v_idx"] == i].drop(columns="v_idx")
        if df_var.empty:
            continue

        merged = df_var.merge(overall_melt, on=["boot_idx", "model"], how="inner").copy()
        merged["dev"] = merged["mean"] - merged["overall_mean"]

        records = []
        for g in all_unique_grps[i]:
            for ms in active_model_shorts:
                sub = merged[(merged["group"] == g) & (merged["model"] == ms)]
                g_devs = sub["dev"].dropna().values
                if len(g_devs) == 0:
                    continue
                lo = float(np.percentile(g_devs, 100.0 * alpha / 2.0))
                hi = float(np.percentile(g_devs, 100.0 * (1.0 - alpha / 2.0)))
                records.append(
                    {
                        "variable":           var,
                        "group":              g,
                        "model":              ms,
                        "mean_dev":           float(np.mean(g_devs)),
                        "ci_lower_dev":       lo,
                        "ci_upper_dev":       hi,
                        "ci_width_dev":       hi - lo,
                        "n_bootstrap":        int(len(g_devs)),
                        "n_respondents_mean": float(sub["n"].mean()),
                    }
                )
        if records:
            pd.DataFrame(records).to_csv(
                deviations_dir / f"deviation_{var}.csv", index=False
            )
            print(f"  Written: deviation_{var}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_double_bootstrap(
    votes_csvs,
    db_path,
    questions_df: pd.DataFrame,
    model_registry: dict    = MODEL_REGISTRY,
    sd_variables: list      = SD_VARIABLES,
    output_dir=None,
    n_bootstrap: int        = N_BOOTSTRAP,
    conf_level: float       = CONFIDENCE_LEVEL,
    seed: int               = RANDOM_SEED,
    n_processes: int | None = None,
    batch_size: int         = BATCH_SIZE,
    question_group_map: dict = QUESTION_GROUP_MAP,
) -> None:
    """Run double bootstrap and write CSV results to *output_dir*.

    Parameters
    ----------
    votes_csvs:
        Iterable of Paths to majority-votes CSVs that contain an
        ``all_answers`` column (generated by Stage 0 majority-vote
        extraction).
    db_path:
        Path to ``ESS11.db`` SQLite database.
    questions_df:
        Question metadata DataFrame produced by ``load_questions``.
        Columns ``var_name``, ``min_scale``, ``max_scale`` are required.
        Questions absent from the majority-votes CSV or the ESS database
        are silently excluded.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir     = output_dir / "scores"
    deviations_dir = output_dir / "deviations"
    scores_dir.mkdir(parents=True, exist_ok=True)
    deviations_dir.mkdir(parents=True, exist_ok=True)

    # ── Load majority-votes CSV (need all_answers column) ─────────────────
    dfs_votes = []
    for p in votes_csvs:
        p = Path(p)
        if p.exists():
            dfs_votes.append(pd.read_csv(p))
        else:
            warnings.warn(f"Votes CSV not found: {p}")
    if not dfs_votes:
        raise FileNotFoundError("No majority-votes CSVs found.")

    votes_full = pd.concat(dfs_votes, ignore_index=True)
    votes_full = votes_full[votes_full["model"].isin(model_registry)].copy()
    if votes_full.empty:
        raise RuntimeError("No registered models found in votes CSVs.")

    if "all_answers" not in votes_full.columns:
        raise RuntimeError(
            "The majority-votes CSV must contain an 'all_answers' column.  "
            "Re-run Stage 0 (extract_majority_votes) to regenerate it."
        )

    # ── Identify active models ─────────────────────────────────────────────
    active_model_fulls  = [
        m for m in model_registry if m in votes_full["model"].unique()
    ]
    active_model_shorts = [model_registry[m] for m in active_model_fulls]
    M = len(active_model_fulls)
    if M == 0:
        raise RuntimeError("No active models found.")

    # ── Load ESS data ──────────────────────────────────────────────────────
    conn   = sqlite3.connect(db_path)
    df_ess = pd.read_sql_query("SELECT * FROM ess11", conn)
    try:
        df_dem = pd.read_sql_query("SELECT * FROM demographics", conn)
        extra_demo = ["new_id"] + [
            c for c in df_dem.columns if c not in df_ess.columns
        ]
        df_ess = df_ess.merge(df_dem[extra_demo], on="new_id", how="left")
    except Exception:
        pass  # demographics table may not exist yet; SD recoding uses ess11 cols
    finally:
        conn.close()

    N = len(df_ess)
    ess_cols = set(df_ess.columns)

    # ── Build question list with scale metadata ────────────────────────────
    q_meta: dict[str, tuple] = {}
    for _, row in questions_df.iterrows():
        var   = str(row["var_name"])
        min_s = float(row["min_scale"])
        max_s = float(row["max_scale"])
        sr    = max_s - min_s
        if sr > 0.0:
            q_meta[var] = (min_s, max_s, sr)

    # Questions must be in scale metadata AND present in the ESS data
    question_vars = [q for q in q_meta if q in ess_cols]
    Q = len(question_vars)
    if Q == 0:
        raise RuntimeError("No questions remain after filtering.")

    print(
        f"  Double bootstrap: {M} models × {Q} questions × "
        f"{N:,} respondents"
    )

    # ── Build answer pool array: (M, Q, pool_size) ────────────────────────
    pool_size   = 20
    answer_pool = np.full((M, Q, pool_size), np.nan, dtype=np.float64)
    for mi, model_full in enumerate(active_model_fulls):
        m_df = (
            votes_full[votes_full["model"] == model_full]
            .set_index("question_var")
        )
        for qi, q in enumerate(question_vars):
            if q in m_df.index:
                answer_pool[mi, qi] = _parse_answers(
                    m_df.loc[q, "all_answers"], expected_size=pool_size
                )

    # ── Verify answer pool ────────────────────────────────────────────────
    _print_pool_diagnostics(answer_pool, active_model_shorts, n_test=200, seed=seed)

    # ── Build respondent answer matrix: (N, Q) ────────────────────────────
    ans_matrix      = np.full((N, Q), np.nan, dtype=np.float64)
    resp_valid_mask = np.zeros((N, Q), dtype=bool)
    scale_ranges    = np.empty(Q, dtype=np.float64)

    for qi, q in enumerate(question_vars):
        min_s, max_s, sr = q_meta[q]
        col = pd.to_numeric(df_ess[q], errors="coerce").values.astype(np.float64)
        ans_matrix[:, qi]      = col
        resp_valid_mask[:, qi] = (
            ~np.isnan(col) & (col >= min_s) & (col <= max_s)
        )
        scale_ranges[qi] = sr

    # ── Question-group indicator matrix: (Q, G) ───────────────────────────
    group_keys   = [question_group_map.get(q, q) for q in question_vars]
    unique_gkeys = list(dict.fromkeys(group_keys))  # insertion-order unique
    g_key_to_id  = {k: i for i, k in enumerate(unique_gkeys)}
    n_groups     = len(unique_gkeys)

    indicator_full = np.zeros((Q, n_groups), dtype=np.float64)
    for qi, gk in enumerate(group_keys):
        indicator_full[qi, g_key_to_id[gk]] = 1.0

    # ── Derive SD variables ────────────────────────────────────────────────
    df_rec = recode_demographics(df_ess)

    all_group_codes: list = []
    all_unique_grps: list = []
    available_vars:  list = []

    for var in sd_variables:
        if var not in df_rec.columns:
            warnings.warn(f"  SD variable not available, skipped: {var}")
            continue
        series        = df_rec[var]
        unique_groups = sorted(series.dropna().unique())
        if not unique_groups:
            warnings.warn(f"  Skipping {var}: no non-null values")
            continue
        g_to_code   = {g: i for i, g in enumerate(unique_groups)}
        group_codes = np.where(
            series.notna(),
            series.map(g_to_code).fillna(-1).astype(int),
            -1,
        ).astype(np.int32)
        all_group_codes.append(group_codes)
        all_unique_grps.append(unique_groups)
        available_vars.append(var)

    # ── Generate per-iteration seeds ──────────────────────────────────────
    rng         = np.random.default_rng(seed)
    model_seeds = rng.integers(0, 2**31, size=n_bootstrap)
    pop_seeds   = rng.integers(0, 2**31, size=n_bootstrap)

    all_iters = [
        (i, int(model_seeds[i]), int(pop_seeds[i]))
        for i in range(n_bootstrap)
    ]
    batches = [
        all_iters[i : i + batch_size]
        for i in range(0, len(all_iters), batch_size)
    ]

    # ── Populate worker globals (inherited by fork, no pickling overhead) ─
    _G.clear()
    _G.update(
        {
            "answer_pool":     answer_pool,
            "ans_matrix":      ans_matrix,
            "resp_valid_mask": resp_valid_mask,
            "scale_ranges":    scale_ranges,
            "indicator_full":  indicator_full,
            "all_group_codes": all_group_codes,
            "all_unique_grps": all_unique_grps,
            "model_shorts":    active_model_shorts,
        }
    )

    # ── Parallel bootstrap ─────────────────────────────────────────────────
    if n_processes is None:
        n_processes = max(1, int(cpu_count() * 0.8))

    print(
        f"  Running {n_bootstrap} double-bootstrap iterations × "
        f"{len(available_vars)} SD variables × {M} models "
        f"[{n_processes} process(es), batch_size={batch_size}]"
    )

    if n_processes > 1:
        with Pool(processes=n_processes) as pool:
            all_raw = pool.map(_dbl_boot_worker, batches)
    else:
        all_raw = [_dbl_boot_worker(b) for b in batches]

    _G.clear()  # release large arrays

    # ── Collect results ────────────────────────────────────────────────────
    flat_groups  = [row for r in all_raw for row in r["groups"]]
    flat_overall = [row for r in all_raw for row in r["overall"]]

    df_boot_all = pd.DataFrame(
        flat_groups,
        columns=["v_idx", "boot_idx", "group", "model", "mean", "n"],
    )
    df_overall_boot = pd.DataFrame(
        flat_overall,
        columns=["boot_idx", "model", "mean", "n"],
    )

    # ── Write overall population CSV ───────────────────────────────────────
    _write_overall(df_overall_boot, active_model_shorts, conf_level, scores_dir)

    # ── Build overall_boot_means: (n_bootstrap, M) ────────────────────────
    # Required by _summarize_and_write_variable → _compute_cross_model_deviation
    overall_pivot = (
        df_overall_boot.pivot_table(
            index="boot_idx", columns="model", values="mean", aggfunc="first"
        )
        .reindex(columns=active_model_shorts)
    )
    # Reindex rows to 0..n_bootstrap-1 so shape is consistent
    overall_pivot = overall_pivot.reindex(range(n_bootstrap))
    overall_boot_means = overall_pivot.to_numpy(dtype=float)  # (n_bootstrap, M)

    # ── Write per-variable summaries ───────────────────────────────────────
    for i, var in enumerate(available_vars):
        print(f"  Summarising → {var}")
        df_var = df_boot_all[df_boot_all["v_idx"] == i].drop(columns="v_idx")
        _summarize_and_write_variable(
            df_var,
            var,
            all_unique_grps[i],
            active_model_shorts,
            conf_level,
            scores_dir,
            overall_boot_means=overall_boot_means,
        )

    # ── Write per-model deviation CSVs ────────────────────────────────────
    print("\n  Writing per-model deviation CSVs …")
    _compute_per_model_deviations(
        df_boot_all,
        df_overall_boot,
        available_vars,
        all_unique_grps,
        active_model_shorts,
        conf_level,
        deviations_dir,
    )

    print(f"\n  Double bootstrap complete → {output_dir}")
