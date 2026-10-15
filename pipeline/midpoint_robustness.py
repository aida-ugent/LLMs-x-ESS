"""
Robustness check: midpoint-answering model.

The synthetic "midpoint" model always answers the exact centre of each
question's Likert scale:

    vote_answer_q = (min_scale_q + max_scale_q) / 2

This serves as a neutral baseline — showing what alignment looks like when
a model never takes any position at all.

Entry point
-----------
run_midpoint_robustness(db_path, question_set_key, questions_df, output_dir, ...)

Steps
-----
1. Compute per-person alignment against the midpoint vote for every question.
2. Write midpoint_{question_set_key}_alignment to the demographics table.
3. Run the standard bootstrap (model_registry = {"midpoint": "midpoint"}).
4. Generate the master overview plot only.

Output
------
  {output_dir}/bootstrap_results/   – bootstrap CSV files
  {output_dir}/figures/master_midpoint.png
"""

import warnings

import numpy as np
import pandas as pd

from .config import (
    QUESTION_GROUP_MAP, SD_VARIABLES,
    N_BOOTSTRAP, CONFIDENCE_LEVEL, RANDOM_SEED, N_PROCESSES, BATCH_SIZE,
)
from .data import write_alignments_to_db
from .bootstrap import run_bootstrap
from .plots import plot_master_overview


_MIDPOINT_REGISTRY: dict[str, str] = {"midpoint": "midpoint"}


# ─────────────────────────────────────────────────────────────────────────────
# Alignment calculation
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_midpoint_alignment(
    df_ess: pd.DataFrame,
    questions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute person-level alignment against the midpoint vote.

    For each question q:
        a_{p,q} = 1 − |answer_{p,q} − midpoint_q| / (max_q − min_q)

    Question-group averaging and person-level mean follow the same logic as
    the main alignment pipeline (see pipeline/alignment.py).

    Returns
    -------
    DataFrame with columns: new_id, midpoint_alignment, n_q_midpoint
    """
    question_vars = [q for q in questions_df["var_name"].tolist() if q in df_ess.columns]

    meta = (
        questions_df[questions_df["var_name"].isin(question_vars)]
        .set_index("var_name")[["min_scale", "max_scale"]]
    )
    meta["min_scale"] = pd.to_numeric(meta["min_scale"], errors="coerce")
    meta["max_scale"] = pd.to_numeric(meta["max_scale"], errors="coerce")
    meta = meta.dropna(subset=["min_scale", "max_scale"])
    meta = meta[meta["max_scale"] > meta["min_scale"]]
    question_vars = [q for q in question_vars if q in meta.index]

    print(f"  Questions available in ESS: {len(question_vars)}")
    print(f"  Respondents:                {len(df_ess):,}")

    new_ids = df_ess["new_id"].values
    N = len(new_ids)

    group_score_lists: dict[str, list] = {}

    for q in question_vars:
        min_s = float(meta.loc[q, "min_scale"])
        max_s = float(meta.loc[q, "max_scale"])
        midpoint = (min_s + max_s) / 2.0
        scale_range = max_s - min_s

        answers = pd.to_numeric(df_ess[q], errors="coerce").values
        valid = (answers >= min_s) & (answers <= max_s) & ~np.isnan(answers)
        pq_scores = np.where(
            valid,
            1.0 - np.abs(answers - midpoint) / scale_range,
            np.nan,
        )

        group_key = QUESTION_GROUP_MAP.get(q, q)
        group_score_lists.setdefault(group_key, []).append(pq_scores)

    if not group_score_lists:
        raise RuntimeError("No questions with valid scales found for midpoint model.")

    # All-NaN rows are masked out by any_valid; silence nanmean's
    # "Mean of empty slice" warning on them.
    group_means: list[np.ndarray] = []
    for arrays in group_score_lists.values():
        stacked = np.column_stack(arrays)
        any_valid = ~np.all(np.isnan(stacked), axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gm = np.nanmean(stacked, axis=1)
        group_means.append(np.where(any_valid, gm, np.nan))

    all_groups = np.column_stack(group_means)
    n_q = np.sum(~np.isnan(all_groups), axis=1).astype(float)
    any_valid = n_q > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        person_align = np.nanmean(all_groups, axis=1)
    person_align = np.where(any_valid, person_align, np.nan)
    n_q = np.where(any_valid, n_q, np.nan)

    print(
        f"  Midpoint alignment: "
        f"{int(np.sum(~np.isnan(person_align))):,} respondents scored"
    )
    return pd.DataFrame({
        "new_id":             new_ids,
        "midpoint_alignment": person_align,
        "n_q_midpoint":       n_q,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_midpoint_robustness(
    db_path,
    question_set_key: str,
    questions_df: pd.DataFrame,
    output_dir,
    figures_dir,
    n_bootstrap: int = N_BOOTSTRAP,
    conf_level: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
    n_processes: int = N_PROCESSES,
    batch_size: int = BATCH_SIZE,
    skip_alignment: bool = False,
    skip_bootstrap: bool = False,
) -> None:
    """
    Run the midpoint robustness check for one question set.

    Parameters
    ----------
    db_path           Path to ESS11.db.
    question_set_key  Key string used in DB column names (e.g. 'all').
    questions_df      Question metadata DataFrame (output of load_questions).
    output_dir        Directory for bootstrap CSV results.
    figures_dir       Directory for the master plot PNG.
    skip_alignment    If True, skip alignment calculation (column already in DB).
    skip_bootstrap    If True, skip bootstrap (use existing CSV files).
    """
    from pathlib import Path
    from .data import load_ess_data

    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    boot_dir = output_dir / "bootstrap_results"
    boot_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Alignment ──────────────────────────────────────────────────
    if not skip_alignment:
        print("\n" + "=" * 60)
        print("MIDPOINT ROBUSTNESS – Alignment calculation")
        print("=" * 60)
        df_ess = load_ess_data(db_path)
        df_person = _calculate_midpoint_alignment(df_ess, questions_df)
        print("\nWriting to database …")
        write_alignments_to_db(
            db_path, df_person, question_set_key,
            model_registry=_MIDPOINT_REGISTRY,
        )
    else:
        print("\n  [skip] Midpoint alignment calculation")

    # ── Stage 2: Bootstrap ──────────────────────────────────────────────────
    if not skip_bootstrap:
        print("\n" + "=" * 60)
        print("MIDPOINT ROBUSTNESS – Bootstrap analysis")
        print("=" * 60)
        run_bootstrap(
            db_path          = db_path,
            question_set_key = question_set_key,
            model_registry   = _MIDPOINT_REGISTRY,
            sd_variables     = SD_VARIABLES,
            output_dir       = boot_dir,
            n_bootstrap      = n_bootstrap,
            conf_level       = conf_level,
            seed             = seed,
            n_processes      = n_processes,
            batch_size       = batch_size,
        )
    else:
        print("\n  [skip] Midpoint bootstrap analysis")

    # ── Stage 3: Master plot ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MIDPOINT ROBUSTNESS – Master overview plot")
    print("=" * 60)
    plot_master_overview(
        bootstrap_results_dir = boot_dir,
        output_dir            = figures_dir,
        model_short           = "midpoint",
        question_set_key      = question_set_key,
    )
    print(f"\n  Plot saved → {figures_dir / 'master_midpoint.png'}")
