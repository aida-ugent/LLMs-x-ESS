"""
Extract majority vote answers from the model-responses interactions database.

Main entry point
----------------
extract_majority_votes(model_responses_db_path, questions_csv, output_dir)
    Reads the interactions table, computes per-model-question majority votes,
    and writes three CSVs to *output_dir*:
      • model_majority_votes.csv          (fed into Stage 1 alignment)
      • model_answer_variation_stats.csv
      • model_refusal_stats.csv
"""

from __future__ import annotations

import sqlite3
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .config import MODEL_REGISTRY, QUESTION_GROUP_MAP


# ── Helpers ────────────────────────────────────────────────────────────────────

def _majority_vote(values: list) -> float:
    """Return the majority vote; break ties by choosing the lower value."""
    vals = [v for v in values if not pd.isna(v)]
    if not vals:
        return np.nan
    counter = Counter(vals)
    max_count = max(counter.values())
    modes = [k for k, v in counter.items() if v == max_count]
    return min(modes)


# ── Core extraction ────────────────────────────────────────────────────────────

def extract_majority_votes(
    model_responses_db_path: Path,
    questions_csv: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Extract majority vote answers from the model-responses interactions database.

    Parameters
    ----------
    model_responses_db_path : Path to model_responses.db (must contain an
                              *interactions* table).
    questions_csv : Path to value_laden_Qs.csv (provides scale metadata).
    output_dir    : Directory where the three output CSVs are written.

    Returns
    -------
    pd.DataFrame  The majority-votes DataFrame (also written to CSV).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_votes    = output_dir / "model_majority_votes.csv"
    output_var      = output_dir / "model_answer_variation_stats.csv"
    output_refusal  = output_dir / "model_refusal_stats.csv"

    print("\n" + "=" * 60)
    print("STAGE 0 – Extract majority votes")
    print("=" * 60)

    # ── Load interactions ──────────────────────────────────────────────────────
    print(f"\n  Connecting to: {model_responses_db_path}")
    conn = sqlite3.connect(model_responses_db_path)
    df = pd.read_sql_query("SELECT * FROM interactions", conn)
    conn.close()

    df = df[df["model_name"].isin(MODEL_REGISTRY)].copy()

    print(f"  Loaded {len(df):,} interactions")
    print(f"  Models    : {df['model_name'].nunique()}")
    print(f"  Questions : {df['survey_id'].nunique()}")
    print(f"  Seeds     : {df['seed'].nunique()}")

    for model in sorted(df["model_name"].unique()):
        n = (df["model_name"] == model).sum()
        print(f"    {model}: {n:,} interactions")

    # ── Load question metadata ─────────────────────────────────────────────────
    df_q = pd.read_csv(questions_csv)
    q_meta: dict[str, dict] = {}
    for _, row in df_q.iterrows():
        var = row["var_name"]
        q_meta[var] = {
            "min_scale": row["min_scale"],
            "max_scale": row["max_scale"],
            "n_response_options": int(row["max_scale"] - row["min_scale"] + 1),
        }
    print(f"\n  Question metadata loaded for {len(q_meta)} variables")

    # ── Numeric answers ────────────────────────────────────────────────────────
    df["answer_num"] = pd.to_numeric(df["answer_code"], errors="coerce")

    votes_rows    = []
    var_rows      = []
    refusal_rows  = []

    for model in sorted(df["model_name"].unique()):
        m_df = df[df["model_name"] == model]

        for question in sorted(m_df["survey_id"].dropna().unique()):
            q_df = m_df[m_df["survey_id"] == question]

            all_responses = q_df["answer_num"].tolist()
            valid         = q_df["answer_num"].dropna().tolist()
            n_total       = len(all_responses)
            n_refusals    = n_total - len(valid)
            pct_refusal   = (n_refusals / n_total * 100) if n_total else 0.0

            refusal_rows.append({
                "model": model,
                "question_var": question,
                "n_total_responses": n_total,
                "n_refusals": n_refusals,
                "n_valid_answers": len(valid),
                "pct_refusal": pct_refusal,
            })

            if not valid:
                continue

            majority = _majority_vote(valid)
            n_disagree = sum(1 for a in valid if a != majority) if not pd.isna(majority) else 0
            pct_disagree = (n_disagree / len(valid) * 100) if valid else 0.0

            meta = q_meta.get(question, {})
            min_s  = meta.get("min_scale",  min(valid) if valid else np.nan)
            max_s  = meta.get("max_scale",  max(valid) if valid else np.nan)
            n_opts = meta.get("n_response_options",
                              int(max_s - min_s + 1) if not pd.isna(max_s) else np.nan)

            seeds = q_df["seed"].dropna().unique()

            votes_rows.append({
                "model": model,
                "question_var": question,
                "majority_vote_answer": majority,
                "n_response_options": n_opts,
                "min_scale": min_s,
                "max_scale": max_s,
                "n_seeds": len(seeds),
                "n_valid_answers": len(valid),
                "answer_variation": len(set(valid)),
                "pct_disagreement": pct_disagree,
                "all_answers": str(sorted(valid)),
            })

            var_rows.append({
                "model": model,
                "question_var": question,
                "n_answers": len(valid),
                "n_disagreements": n_disagree,
                "pct_disagreement": pct_disagree,
            })

    df_votes   = pd.DataFrame(votes_rows)
    df_var     = pd.DataFrame(var_rows)
    df_refusal = pd.DataFrame(refusal_rows)

    print(f"\n  Majority votes computed: {len(df_votes)} model-question pairs")

    # ── Refusal summary ────────────────────────────────────────────────────────
    _print_refusal_summary(df_refusal)

    # ── Variation summary ──────────────────────────────────────────────────────
    df_var_agg = _aggregate_variation(df_var)
    _print_variation_summary(df_var_agg)

    # ── Seed agreement ─────────────────────────────────────────────────────────
    _print_seed_agreement(df_votes)

    # ── Save ───────────────────────────────────────────────────────────────────
    df_votes.to_csv(output_votes, index=False)
    df_var_agg.to_csv(output_var, index=False)
    df_refusal.to_csv(output_refusal, index=False)  # per-question refusal detail

    print(f"\n  Saved: {output_votes}")
    print(f"  Saved: {output_var}")
    print(f"  Saved: {output_refusal}")

    return df_votes


# ── Reporting helpers ──────────────────────────────────────────────────────────

def _print_refusal_summary(df_refusal: pd.DataFrame) -> None:
    if df_refusal.empty:
        return
    print("\n  Refusal summary by model:")
    for model in sorted(df_refusal["model"].unique()):
        md = df_refusal[df_refusal["model"] == model]
        total_r   = md["n_total_responses"].sum()
        total_ref = md["n_refusals"].sum()
        pct       = total_ref / total_r * 100 if total_r else 0
        n_all_ref = (md["n_valid_answers"] == 0).sum()
        print(f"    {model}: overall refusal {pct:.1f}%"
              f"  ({total_ref}/{total_r}),"
              f"  {n_all_ref} questions with 0 valid answers")


def _aggregate_variation(df_var: pd.DataFrame) -> pd.DataFrame:
    if df_var.empty:
        return df_var
    df_var = df_var.copy()
    df_var["question_group"] = df_var["question_var"].map(
        lambda q: QUESTION_GROUP_MAP.get(q, q)
    )
    grp = df_var.groupby(["model", "question_group"]).agg(
        n_answers=("n_answers", "sum"),
        n_disagreements=("n_disagreements", "sum"),
    ).reset_index()
    grp["pct_disagreement"] = (
        grp["n_disagreements"] / grp["n_answers"] * 100
    ).fillna(0)

    summary_rows = []
    for model in sorted(grp["model"].unique()):
        md = grp[grp["model"] == model]
        n_with_var = (md["pct_disagreement"] > 0).sum()
        summary_rows.append({
            "model": model,
            "n_questions": len(md),
            "mean_pct_disagreement_per_question": md["pct_disagreement"].mean(),
            "median_pct_disagreement_per_question": md["pct_disagreement"].median(),
            "std_pct_disagreement_per_question": md["pct_disagreement"].std(),
            "n_questions_with_variation": int(n_with_var),
            "pct_questions_with_variation": n_with_var / len(md) * 100 if len(md) else 0,
            "n_questions_perfect_agreement": int(len(md) - n_with_var),
            "pct_questions_perfect_agreement": (len(md) - n_with_var) / len(md) * 100 if len(md) else 0,
        })
    return pd.DataFrame(summary_rows)


def _print_variation_summary(df_var_agg: pd.DataFrame) -> None:
    if df_var_agg.empty:
        return
    print("\n  Answer variation (mean % disagreement with majority per question):")
    for _, row in df_var_agg.iterrows():
        print(f"    {row['model']}: {row['mean_pct_disagreement_per_question']:.2f}%"
              f"  ({row['n_questions_with_variation']}/{row['n_questions']} questions have variation)")


def _print_seed_agreement(df_votes: pd.DataFrame) -> None:
    if df_votes.empty:
        return
    print("\n  Seed agreement summary:")
    for model in sorted(df_votes["model"].unique()):
        md = df_votes[df_votes["model"] == model]
        perfect = (md["answer_variation"] == 1).sum()
        pct     = perfect / len(md) * 100
        print(f"    {model}: {perfect}/{len(md)} questions perfect agreement ({pct:.1f}%)")
