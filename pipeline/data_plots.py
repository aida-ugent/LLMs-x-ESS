"""
data_plots.py – Descriptive figures and tables about the survey data and model
behaviour.  Everything is written into the 'data_plots' folder passed as
*output_dir*.

Non-answer / refusal
--------------------
  people_nonanswer_per_question.{png,csv}
        Horizontal bars: % non-answer per question for ESS11 respondents,
        coloured by topic.
  people_nonanswer_per_topic.{png,csv}
        The same rates averaged per topic.
  people_nonanswer_combined.png
        Per-question bars with the per-topic chart as an inset.
  model_refusal_per_question.{png,csv}
        Heatmap – rows = models, columns = questions ordered most-refused
        first.  Fill: white (0 % refusal) → red (100 % refusal).
  model_response_heatmap.png
        Heatmap – rows = models, columns = questions ordered fewest→most
        total valid responses.  Fill: dark green (0 % answered) → white
        (100 % answered).  Questions where ≥1 model gave 0 valid answers get
        a thick black border and a hatched overlay.
  people_questions_answered_hist.{png,csv}
        Bars: how many ESS11 respondents answered exactly N questions
        (out of the full 'all' question set).

Answer distribution
-------------------
  avg_answer_per_question.{png,csv}, avg_answer_per_question_tall.png
        Mean normalised (0–1) answer per question for the ESS11 population
        and each model, in a wide and a tall layout.

Method comparison
-----------------
  method_comp_r2_by_llm_all_vs_just_values_v2.png
        R² by LLM for the 'all' and 'just_values' question sets in one figure.
        Skipped when the method-comparison CSVs do not exist yet.

Tables
------
  missing_data_per_sociodemographic.{csv,tex}
        Missing-data counts per sociodemographic variable.
  respondents_by_country.tex
        Respondents per country.

Entry point
-----------
    run_data_plots(db_path, refusal_stats_csv, questions_csv, output_dir,
                   majority_votes_csvs)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import colorsys
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from .config import MODEL_REGISTRY, QUESTION_GROUP_MAP, COLOR_PALETTE

# ── Colour helpers ─────────────────────────────────────────────────────────────

_TOPIC_PALETTE = COLOR_PALETTE


def _topic_color_map(topics) -> dict[str, str]:
    sorted_topics = sorted(set(topics))
    return {t: _TOPIC_PALETTE[i % len(_TOPIC_PALETTE)] for i, t in enumerate(sorted_topics)}


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_all_questions(questions_csv: Path) -> pd.DataFrame:
    """Load the 'all' question set (has_question == True)."""
    df = pd.read_csv(questions_csv)
    df["has_question"] = df["has_question"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    return df[df["has_question"]].reset_index(drop=True)


def _load_ess_question_cols(db_path: Path, questions_df: pd.DataFrame) -> pd.DataFrame:
    """Load only the question columns from ess11."""
    vars_ = [v for v in questions_df["var_name"].tolist()]
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT {', '.join(vars_)} FROM ess11", conn)
    conn.close()
    return df


def _valid_mask(df_ess: pd.DataFrame, questions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Boolean DataFrame – True where a respondent gave a valid answer.
    Valid = numeric value within [min_scale, max_scale].
    """
    masks = {}
    for _, row in questions_df.iterrows():
        var = row["var_name"]
        if var not in df_ess.columns:
            continue
        col = pd.to_numeric(df_ess[var], errors="coerce")
        masks[var] = col.between(float(row["min_scale"]), float(row["max_scale"]))
    return pd.DataFrame(masks)


# ── Plot 1: People non-answer rate per question ────────────────────────────────

def plot_people_nonanswer(
    db_path: Path,
    questions_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Horizontal bar chart of % non-answer per question for ESS11 respondents.
    Grouped questions (testjc*) are collapsed per QUESTION_GROUP_MAP; questions
    appear on the y-axis, coloured by topic."""
    df_ess = _load_ess_question_cols(db_path, questions_df)
    mask = _valid_mask(df_ess, questions_df)

    # Collapse grouped questions into single entries
    grouped_vars = set(QUESTION_GROUP_MAP.keys())
    seen_groups: set[str] = set()
    entries: list[dict] = []
    for _, row in questions_df.iterrows():
        var = str(row["var_name"])
        topic = str(row["topic"]).strip()
        if var in grouped_vars:
            grp = QUESTION_GROUP_MAP[var]
            if grp in seen_groups:
                continue
            seen_groups.add(grp)
            members = [v for v, g in QUESTION_GROUP_MAP.items() if g == grp]
            entries.append({"display_name": grp, "topic": topic, "members": members})
        else:
            entries.append({"display_name": var, "topic": topic, "members": [var]})

    rows = []
    for entry in entries:
        avail = [m for m in entry["members"] if m in mask.columns]
        if not avail:
            continue
        answered = mask[avail].any(axis=1) if len(avail) > 1 else mask[avail[0]]
        pct_na = (~answered).mean() * 100
        rows.append({
            "display_name": entry["display_name"],
            "topic":        entry["topic"],
            "pct_nonanswer": pct_na,
        })
    # Sort: most non-answers at the top (index 0 after invert_yaxis)
    df = pd.DataFrame(rows).sort_values("pct_nonanswer", ascending=False).reset_index(drop=True)

    topic_cols = _topic_color_map(df["topic"])
    bar_colors = [topic_cols[t] for t in df["topic"]]

    n = len(df)
    fig_h = max(8, n * 0.30)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y = np.arange(n)
    bars = ax.barh(y, df["pct_nonanswer"], color=bar_colors,
                   edgecolor="white", linewidth=0.3, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(df["display_name"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("% Non-answer (refusal / don't know / out-of-range / missing)")
    ax.set_title("ESS11 respondents — non-answer rate per question (all question set)")

    # Add percentage label at the end of each bar
    x_max = df["pct_nonanswer"].max()
    ax.set_xlim(left=0, right=x_max * 1.12)
    for bar, pct in zip(bars, df["pct_nonanswer"]):
        ax.text(
            bar.get_width() + x_max * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", ha="left", fontsize=6.5,
        )

    legend_patches = [
        mpatches.Patch(color=col, label=topic)
        for topic, col in sorted(topic_cols.items())
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=8, title="Topic", framealpha=0.9)

    fig.tight_layout()
    out_png = output_dir / "people_nonanswer_per_question.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png} (+ .pdf)")

    out_csv = output_dir / "people_nonanswer_per_question.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")


# ── Plot 2: Model refusal rate per question ────────────────────────────────────

def plot_model_refusal_per_question(
    refusal_stats_csv: Path,
    questions_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Heatmap: rows = models, columns = questions ordered most-refused first.
    Fill colour: white (0 % refusal) → red (100 % refusal).
    Black border on questions where ≥1 model gave 0 valid answers.
    """
    df_refusal = pd.read_csv(refusal_stats_csv)
    df_refusal = df_refusal[df_refusal["model"].isin(MODEL_REGISTRY)].copy()

    q_vars = questions_df["var_name"].tolist()
    df_refusal = df_refusal[df_refusal["question_var"].isin(q_vars)].copy()

    df_refusal["model_short"] = df_refusal["model"].map(MODEL_REGISTRY)
    model_shorts = sorted(df_refusal["model_short"].dropna().unique())

    df_pivot = df_refusal.pivot_table(
        index="model_short", columns="question_var",
        values="pct_refusal", aggfunc="mean",
    ).reindex(index=model_shorts)

    # Order columns: most total refusal first
    col_totals = df_pivot.sum(axis=0).sort_values(ascending=False)
    q_order = col_totals.index.tolist()
    df_pivot = df_pivot[q_order]

    # Questions where ≥1 model gave 0 valid answers
    zero_qs = set(
        df_refusal.loc[df_refusal["n_valid_answers"] == 0, "question_var"].unique()
    )

    n_models    = len(model_shorts)
    n_questions = len(q_order)

    cmap = LinearSegmentedColormap.from_list(
        "white_red", ["#ffffff", "#d62728"], N=256
    )

    fig_w = max(16, n_questions * 0.38)
    fig_h = max(4,  n_models   * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mat = df_pivot.values / 100.0   # normalise to 0–1
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="none")

    for j, q in enumerate(q_order):
        if q in zero_qs:
            for i in range(n_models):
                hatch_rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    linewidth=0, edgecolor="#888888", facecolor="none",
                    hatch="////", zorder=4,
                )
                ax.add_patch(hatch_rect)
            border = mpatches.Rectangle(
                (j - 0.5, -0.5), 1.0, n_models,
                linewidth=2.5, edgecolor="black", facecolor="none", zorder=6,
            )
            ax.add_patch(border)

    ax.set_xticks(range(n_questions))
    ax.set_xticklabels(q_order, rotation=90, fontsize=6)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_shorts, fontsize=8)
    ax.set_title(
        "Model refusal rate per question  (white = 0 % → red = 100 % refusal)\n"
        "Black border + hatch: ≥ 1 model gave 0 valid answers — ordered most refused first"
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cbar.set_label("Refusal rate", fontsize=8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])

    fig.tight_layout()
    out_png = output_dir / "model_refusal_per_question.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png} (+ .pdf)")

    out_csv = output_dir / "model_refusal_per_question.csv"
    df_pivot.to_csv(out_csv)
    print(f"  Saved: {out_csv}")


# ── Plot 3: Model response heatmap ─────────────────────────────────────────────

def plot_model_response_heatmap(
    refusal_stats_csv: Path,
    questions_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Heatmap: rows = models, columns = questions.
    - Questions ordered by total valid responses across all models (fewest first).
    - Fill colour: dark green (0 % answered) → white (100 % answered).
    - Thick black border and a hatched overlay on questions where ≥1 model
      gave 0 valid answers.
    """
    df_refusal = pd.read_csv(refusal_stats_csv)
    df_refusal = df_refusal[df_refusal["model"].isin(MODEL_REGISTRY)].copy()

    q_vars = questions_df["var_name"].tolist()
    df_refusal = df_refusal[df_refusal["question_var"].isin(q_vars)].copy()

    df_refusal["response_rate"] = (
        df_refusal["n_valid_answers"] / df_refusal["n_total_responses"]
    ).fillna(0.0).clip(0, 1)

    # Total valid responses per question (across all models) → sort order
    q_total = (
        df_refusal.groupby("question_var")["n_valid_answers"]
        .sum()
        .reindex(q_vars)
        .fillna(0)
    )
    # Fewest responses first
    q_order = q_total.sort_values(ascending=True).index.tolist()

    # Questions where ≥1 model gave 0 valid answers
    zero_qs = set(
        df_refusal.loc[df_refusal["n_valid_answers"] == 0, "question_var"].unique()
    )

    df_refusal["model_short"] = df_refusal["model"].map(MODEL_REGISTRY)
    model_shorts = sorted(df_refusal["model_short"].dropna().unique())

    df_pivot = df_refusal.pivot_table(
        index="model_short", columns="question_var",
        values="response_rate", aggfunc="mean",
    ).reindex(index=model_shorts, columns=q_order)

    n_models = len(model_shorts)
    n_questions = len(q_order)

    # Custom colormap: dark green (0 % answered) → white (100 % answered)
    cmap = LinearSegmentedColormap.from_list(
        "green_white", ["#004529", "#ffffff"], N=256
    )

    fig_w = max(16, n_questions * 0.38)
    fig_h = max(4,  n_models   * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mat = df_pivot.values  # (n_models, n_questions)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="none")

    # Flagged columns: gray diagonal-hatch overlay per cell + thick black column border
    for j, q in enumerate(q_order):
        if q in zero_qs:
            # Gray hatching over every individual cell in this column
            for i in range(n_models):
                hatch_rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    linewidth=0, edgecolor="#888888", facecolor="none",
                    hatch="////", zorder=4,
                )
                ax.add_patch(hatch_rect)
            # Thick black border around the whole column on top
            border = mpatches.Rectangle(
                (j - 0.5, -0.5), 1.0, n_models,
                linewidth=2.5, edgecolor="black", facecolor="none", zorder=6,
            )
            ax.add_patch(border)

    ax.set_xticks(range(n_questions))
    ax.set_xticklabels(q_order, rotation=90, fontsize=6)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_shorts, fontsize=8)
    ax.set_title(
        "Model response rate per question  "
        "(dark green = never answered → white = always answered)\n"
        "Black border + hatch: ≥ 1 model gave 0 valid answers — "
        "questions ordered fewest → most total responses"
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cbar.set_label("Response rate (valid / total seeds)", fontsize=8)

    fig.tight_layout()
    out_png = output_dir / "model_response_heatmap.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png} (+ .pdf)")


# ── Plot 4: People questions answered histogram ────────────────────────────────

def plot_people_questions_answered(
    db_path: Path,
    questions_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Bar chart: number of ESS11 respondents who answered exactly N questions
    (grouped questions per QUESTION_GROUP_MAP count as 1, max = 47).
    Dashed lines show the mean and median.
    """
    df_ess = _load_ess_question_cols(db_path, questions_df)
    mask = _valid_mask(df_ess, questions_df)

    # Individual questions (not part of any group)
    grouped_vars = set(QUESTION_GROUP_MAP.keys())
    individual_vars = [c for c in mask.columns if c not in grouped_vars]

    # Groups: count 1 if any member answered
    groups: dict[str, list[str]] = {}
    for var, grp in QUESTION_GROUP_MAP.items():
        groups.setdefault(grp, []).append(var)

    individual_count = (
        mask[individual_vars].sum(axis=1)
        if individual_vars
        else pd.Series(0, index=mask.index)
    )
    group_count = pd.Series(0, index=mask.index, dtype=int)
    for grp, members in groups.items():
        avail = [m for m in members if m in mask.columns]
        if avail:
            group_count = group_count + mask[avail].any(axis=1).astype(int)

    n_answered  = (individual_count + group_count).clip(upper=47)
    counts      = n_answered.value_counts().sort_index()
    n_respondents = len(n_answered)

    mean_val   = n_answered.mean()
    median_val = n_answered.median()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(counts.index.astype(int), counts.values,
           color="#0173B2", edgecolor="white", linewidth=0.3)

    ax.axvline(mean_val,   color="#D55E00", linestyle="--", linewidth=1.8,
               label=f"Mean: {mean_val:.1f}", zorder=5)
    ax.axvline(median_val, color="#029E73", linestyle="--", linewidth=1.8,
               label=f"Median: {int(median_val)}", zorder=5)
    ax.legend(fontsize=9, framealpha=0.9)

    ax.set_xlabel("Number of questions answered (grouped, max 47)")
    ax.set_ylabel("Number of respondents")
    ax.set_title(
        f"Distribution of questions answered per respondent\n"
        f"(ESS11, all question set, N = {n_respondents:,})"
    )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlim(left=-0.5, right=47.5)

    fig.tight_layout()
    out_png = output_dir / "people_questions_answered_hist.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png} (+ .pdf)")

    out_csv = output_dir / "people_questions_answered_hist.csv"
    pd.DataFrame({
        "n_questions_answered": counts.index.astype(int),
        "n_respondents":        counts.values,
    }).to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")


# ── Plot 5: People non-answer rate per topic ──────────────────────────────────

def plot_people_nonanswer_per_topic(
    db_path: Path,
    questions_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Horizontal bar chart of mean % non-answer aggregated by topic.
    Each bar is the average non-answer rate across questions in that topic
    (grouped questions collapsed per QUESTION_GROUP_MAP count as one).
    Bars are coloured by topic using the same palette as plot 1, sorted
    so the topic with the most non-answers is at the top.
    """
    df_ess = _load_ess_question_cols(db_path, questions_df)
    mask = _valid_mask(df_ess, questions_df)

    # Collapse grouped questions, same logic as plot 1
    grouped_vars = set(QUESTION_GROUP_MAP.keys())
    seen_groups: set[str] = set()
    entries: list[dict] = []
    for _, row in questions_df.iterrows():
        var = str(row["var_name"])
        topic = str(row["topic"]).strip()
        if var in grouped_vars:
            grp = QUESTION_GROUP_MAP[var]
            if grp in seen_groups:
                continue
            seen_groups.add(grp)
            members = [v for v, g in QUESTION_GROUP_MAP.items() if g == grp]
            entries.append({"display_name": grp, "topic": topic, "members": members})
        else:
            entries.append({"display_name": var, "topic": topic, "members": [var]})

    rows = []
    for entry in entries:
        avail = [m for m in entry["members"] if m in mask.columns]
        if not avail:
            continue
        answered = mask[avail].any(axis=1) if len(avail) > 1 else mask[avail[0]]
        pct_na = (~answered).mean() * 100
        rows.append({"topic": entry["topic"], "pct_nonanswer": pct_na})

    df_q = pd.DataFrame(rows)
    df_topic = (
        df_q.groupby("topic")["pct_nonanswer"]
        .agg(mean_pct="mean", n_questions="count")
        .reset_index()
        .sort_values("mean_pct", ascending=False)  # most non-answers at top after invert
        .reset_index(drop=True)
    )

    topic_cols = _topic_color_map(df_topic["topic"])
    bar_colors = [topic_cols[t] for t in df_topic["topic"]]

    n = len(df_topic)
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.55)))

    y = np.arange(n)
    bars = ax.barh(y, df_topic["mean_pct"], color=bar_colors,
                   edgecolor="white", linewidth=0.4, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{t}  (n={int(nq)})" for t, nq in zip(df_topic["topic"], df_topic["n_questions"])],
        fontsize=9,
    )
    ax.invert_yaxis()

    x_max = df_topic["mean_pct"].max()
    ax.set_xlim(left=0, right=x_max * 1.15)
    for bar, pct in zip(bars, df_topic["mean_pct"]):
        ax.text(
            bar.get_width() + x_max * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", ha="left", fontsize=8.5,
        )

    ax.set_xlabel("Mean % non-answer across questions in topic")
    ax.set_title("ESS11 respondents — mean non-answer rate per topic (all question set)")

    fig.tight_layout()
    out_png = output_dir / "people_nonanswer_per_topic.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png} (+ .pdf)")

    out_csv = output_dir / "people_nonanswer_per_topic.csv"
    df_topic.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")


# ── Plot combined: per-question with per-topic inset ──────────────────────────

def plot_people_nonanswer_combined(
    db_path: Path,
    questions_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Per-question horizontal bar chart with a small per-topic inset axes
    embedded in the bottom-right whitespace.
    """
    df_ess = _load_ess_question_cols(db_path, questions_df)
    mask   = _valid_mask(df_ess, questions_df)

    grouped_vars = set(QUESTION_GROUP_MAP.keys())
    seen_groups: set[str] = set()
    entries: list[dict] = []
    for _, row in questions_df.iterrows():
        var   = str(row["var_name"])
        topic = str(row["topic"]).strip()
        if var in grouped_vars:
            grp = QUESTION_GROUP_MAP[var]
            if grp in seen_groups:
                continue
            seen_groups.add(grp)
            members = [v for v, g in QUESTION_GROUP_MAP.items() if g == grp]
            entries.append({"display_name": grp, "topic": topic, "members": members})
        else:
            entries.append({"display_name": var, "topic": topic, "members": [var]})

    rows = []
    for entry in entries:
        avail = [m for m in entry["members"] if m in mask.columns]
        if not avail:
            continue
        answered = mask[avail].any(axis=1) if len(avail) > 1 else mask[avail[0]]
        pct_na   = (~answered).mean() * 100
        rows.append({
            "display_name": entry["display_name"],
            "topic":        entry["topic"],
            "pct_nonanswer": pct_na,
        })

    df_q = (
        pd.DataFrame(rows)
        .sort_values("pct_nonanswer", ascending=False)
        .reset_index(drop=True)
    )
    df_topic = (
        df_q.groupby("topic")["pct_nonanswer"]
        .agg(mean_pct="mean", n_questions="count")
        .reset_index()
        .sort_values("mean_pct", ascending=False)
        .reset_index(drop=True)
    )

    topic_cols = _topic_color_map(df_q["topic"])

    # ── Main axes: per-question (vertical bars, wide layout) ──────────────────
    n_q   = len(df_q)
    fig_w = max(16, n_q * 0.30)
    fig, ax = plt.subplots(figsize=(fig_w, 10))

    bar_colors_q = [topic_cols[t] for t in df_q["topic"]]
    xq   = np.arange(n_q)
    bars = ax.bar(xq, df_q["pct_nonanswer"], color=bar_colors_q,
                  edgecolor="white", linewidth=0.3, width=0.72)
    ax.set_xticks(xq)
    ax.set_xticklabels(df_q["display_name"], rotation=90, ha="center", fontsize=9.5)
    # Pin x-limits tightly around the bars
    ax.set_xlim(-0.5, n_q - 0.5)
    y_max_q = df_q["pct_nonanswer"].max()
    ax.set_ylim(bottom=0, top=y_max_q * 1.18)
    for bar, pct in zip(bars, df_q["pct_nonanswer"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max_q * 0.01,
            f"{pct:.1f}%",
            va="bottom", ha="center", fontsize=9.5, rotation=90,
        )
    ax.set_ylabel("% Non-answer (refusal / don't know / out-of-range / missing)", fontsize=12)
    ax.set_title(
        "ESS11 respondents — non-answer rate per question\n"
        "Bars coloured by topic; inset shows mean rate aggregated by topic",
        fontsize=12, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=11)

    # ── Inset axes: per-topic (upper-right whitespace) ─────────────────────────
    ax_inset = ax.inset_axes([0.68, 0.52, 0.30, 0.45])

    n_t = len(df_topic)
    bar_colors_t = [topic_cols[t] for t in df_topic["topic"]]
    yt = np.arange(n_t)
    bars_t = ax_inset.barh(yt, df_topic["mean_pct"], color=bar_colors_t,
                           edgecolor="white", linewidth=0.3, height=0.65)
    ax_inset.set_yticks(yt)
    ax_inset.set_yticklabels(
        [f"{t}  (n={int(nq)})"
         for t, nq in zip(df_topic["topic"], df_topic["n_questions"])],
        fontsize=9,
    )
    ax_inset.invert_yaxis()
    ax_inset.set_ylim(n_t - 0.5, -0.5)
    x_max_t = df_topic["mean_pct"].max()
    ax_inset.set_xlim(left=0, right=x_max_t * 1.32)
    for bar, pct in zip(bars_t, df_topic["mean_pct"]):
        ax_inset.text(
            bar.get_width() + x_max_t * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", ha="left", fontsize=8.5,
        )
    ax_inset.set_xlabel("")
    ax_inset.set_title("By topic  (mean % non-answer)", fontsize=10, fontweight="bold")
    ax_inset.tick_params(axis="both", labelsize=8.5)

    # Legend inside the inset
    legend_patches = [
        mpatches.Patch(color=col, label=topic)
        for topic, col in sorted(topic_cols.items())
    ]
    ax_inset.legend(handles=legend_patches, loc="lower right",
                    fontsize=10, title="Topic", title_fontsize=10,
                    framealpha=0.9)

    # Light border around the inset
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("#aaaaaa")
        spine.set_linewidth(0.8)

    fig.tight_layout()
    out_png = output_dir / "people_nonanswer_combined.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png} (+ .pdf)")


# ── Plot 6: Combined method-comp R² by LLM ────────────────────────────────────

_MC_MODEL_ORDER = [
    "Country-only (Linear)",
    "SD-only (Linear)",
    "SD-only (GBM)",
    "Country+SD (Linear)",
    "Country+SD (GBM)",
]


def _load_mc_results(mc_dir: Path) -> list[dict]:
    """Read all method_comp_<model>.csv files and return a flat list of dicts."""
    rows = []
    for csv_path in sorted(mc_dir.glob("method_comp_*.csv")):
        if "all_models" in csv_path.stem:
            continue
        try:
            df = pd.read_csv(csv_path)
            for col in ("model", "r2_mean", "r2_std", "llm_model"):
                if col not in df.columns:
                    break
            else:
                for _, row in df.iterrows():
                    rows.append({
                        "method":    row["model"],
                        "r2_mean":   row["r2_mean"],
                        "r2_std":    row["r2_std"],
                        "llm_model": row["llm_model"],
                    })
        except Exception:
            pass
    return rows


def _draw_r2_by_llm_ax(
    ax,
    all_results: list[dict],
    title: str,
    weight_col: str,
    show_legend: bool,
    ylim=None,
    legend_style: str = "box",
) -> list:
    """
    Draw a single R²-by-LLM grouped bar chart onto *ax*. Returns legend handles.
    legend_style: 'box'  – framed legend in upper-right corner
                  'free' – frameless, top/right spines removed, legend under title
    """
    from .config import COLOR_PALETTE

    if not all_results:
        ax.set_visible(False)
        return []

    df = pd.DataFrame(all_results)
    llm_models = sorted([m for m in df["llm_model"].unique() if m in set(MODEL_REGISTRY.values())])
    n_llm     = len(llm_models)
    n_methods = len(_MC_MODEL_ORDER)
    method_colors = dict(zip(_MC_MODEL_ORDER, COLOR_PALETTE[:n_methods]))

    def _darken(hex_color: str, factor: float = 0.55) -> str:
        """Return a darkened version of a hex colour (factor < 1 = darker)."""
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        r2, g2, b2 = colorsys.hls_to_rgb(h, max(0, l * factor), s)
        return "#{:02x}{:02x}{:02x}".format(int(r2*255), int(g2*255), int(b2*255))

    total_width = 0.92
    bar_w = total_width / n_methods
    x = np.arange(n_llm)

    handles = []
    for i, method in enumerate(_MC_MODEL_ORDER):
        sub = (
            df[df["method"] == method]
            .set_index("llm_model")
            .reindex(llm_models)
        )
        offsets = x - total_width / 2 + bar_w * (i + 0.5)
        bars = ax.bar(
            offsets,
            sub["r2_mean"],
            width=bar_w * 0.96,
            yerr=sub["r2_std"],
            capsize=4,
            alpha=0.85,
            edgecolor=_darken(method_colors[method]),
            linewidth=0.8,
            color=method_colors[method],
            label=method,
        )
        handles.append(bars[0])
        for rect, (mv_raw, sv_raw) in zip(bars, zip(sub["r2_mean"], sub["r2_std"])):
            try:
                mv = float(mv_raw)
                sv = float(sv_raw) if not np.isnan(float(sv_raw)) else 0.0
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    mv / 2,
                    f"{mv:.3f}",
                    ha="center", va="center",
                    fontsize=10, rotation=90,
                    color="white", fontweight="bold",
                )
            except (TypeError, ValueError):
                pass

    ax.set_xlim(-0.48, n_llm - 0.52)
    ax.set_xticks(x)
    ax.set_xticklabels(llm_models, rotation=30, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel(f"Weighted Test R\u00b2 ({weight_col}, 10-fold CV)",
                  fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold",
                 pad=8 if legend_style != "free" else 32)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_axisbelow(True)

    if legend_style == "free":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if show_legend:
        if legend_style == "free":
            ax.legend(handles=handles, labels=_MC_MODEL_ORDER,
                      fontsize=13, frameon=False, ncol=len(_MC_MODEL_ORDER),
                      loc="lower center",
                      bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3)
        else:
            ax.legend(handles=handles, labels=_MC_MODEL_ORDER,
                      fontsize=13, loc="upper right", borderaxespad=0.5,
                      framealpha=0.9)
    return handles


def plot_method_comp_r2_by_llm_combined(
    results_dir: Path,
    output_dir: Path,
) -> None:
    """
    Draw method_comp R²-by-LLM for 'all' (top) and 'just_values' (bottom)
    in a single native figure.  Single legend shown once in the top panel.
    Skipped silently if either method_comp directory has no CSVs yet.
    """
    top_mc_dir = results_dir / "all_answered" / "method_comp"
    jv_mc_dir  = results_dir / "just_values"  / "method_comp"

    if not top_mc_dir.exists() or not jv_mc_dir.exists():
        print("  Skipping combined R\u00b2 plot \u2014 run --run-method-comp for both question sets first.")
        return

    top_results = _load_mc_results(top_mc_dir)
    jv_results  = _load_mc_results(jv_mc_dir)

    if not top_results or not jv_results:
        print("  Skipping combined R\u00b2 plot \u2014 no method_comp CSVs found.")
        return

    # Determine weight_col from first row
    weight_col = "pspwght"
    try:
        import csv
        with open(next(top_mc_dir.glob("method_comp_*.csv"))) as f:
            reader = csv.DictReader(f)
            row = next(reader, {})
            weight_col = row.get("weight_col", weight_col)
    except Exception:
        pass

    n_llm = len(set(r["llm_model"] for r in top_results + jv_results
                    if r["llm_model"] in set(MODEL_REGISTRY.values())))
    fig_w = max(12, 1.8 * n_llm)

    # Shared y-axis limits across both panels
    combined = pd.DataFrame(top_results + jv_results)
    y_min = (combined["r2_mean"] - combined["r2_std"].fillna(0)).min()
    y_max = (combined["r2_mean"] + combined["r2_std"].fillna(0)).max()
    margin = (y_max - y_min) * 0.08
    shared_ylim = (y_min - margin, y_max + margin)

    # ── v2: frameless, top/right spines removed, legend under title ──────────
    fig2, axes2 = plt.subplots(2, 1, figsize=(fig_w, 10),
                               constrained_layout=True)
    _draw_r2_by_llm_ax(axes2[0], top_results,
                        r"Answered by All LLMS ($\mathcal{Q}$)", weight_col, show_legend=True,
                        ylim=shared_ylim, legend_style="free")
    _draw_r2_by_llm_ax(axes2[1], jv_results,
                        "Reduced Portrait Value Questionnaire",  weight_col, show_legend=True,
                        ylim=shared_ylim, legend_style="free")

    out_path2 = output_dir / "method_comp_r2_by_llm_all_vs_just_values_v2.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    fig2.savefig(out_path2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out_path2} (+ .pdf)")


# ── Table: Missing data per master-plot sociodemographic variable ──────────────

# Metadata for the 15 master-plot sociodemographic variables.
# 'missing_codes' describes the ESS raw codes that are recoded to NaN
# (i.e., what is considered missing in the analysis).
# Note: for religion_clean, 'No religion' (rlgblg=2 → 0.0) is NOT missing.
_MASTER_VAR_META = [
    {
        "var":          "gender_clean",
        "label":        "Gender",
        "source":       "nobingnd",
        "missing_codes": "Values outside {1=Man, 2=Woman, 3=Non-binary}; typically 9=No answer",
    },
    {
        "var":          "ethnicmajority_clean",
        "label":        "Ethnic Majority",
        "source":       "feethngr",
        "missing_codes": "Values outside {1=Belong to majority, 2=Do not belong}; 7=Refusal, 8=Don't know, 9=No answer",
    },
    {
        "var":          "immigration_background",
        "label":        "Immigration Background",
        "source":       "brncntr, facntr, mocntr (+ cntbrthd, fbrncntc, mbrncntc)",
        "missing_codes": "brncntr in {7=Refusal, 8=Don't know, 9=No answer}; or insufficient parent-country information to classify",
    },
    {
        "var":          "education_bracket",
        "label":        "Education (ISCED)",
        "source":       "edulvlb",
        "missing_codes": "5555=Other, 7777=Refusal, 8888=Don't know, 9999=No answer",
    },
    {
        "var":          "income_clean",
        "label":        "Income Decile",
        "source":       "hinctnta",
        "missing_codes": "Values outside 1–10; 77=Refusal, 88=Don't know, 99=No answer",
    },
    {
        "var":          "hincfel_clean",
        "label":        "Household Income Feeling",
        "source":       "hincfel",
        "missing_codes": "Values outside {1–4}; 7=Refusal, 8=Don't know, 9=No answer",
    },
    {
        "var":          "fnsdfml_clean",
        "label":        "Childhood Financial Difficulties",
        "source":       "fnsdfml",
        "missing_codes": "Values outside {1–5}; 7=Refusal, 8=Don't know, 9=No answer",
    },
    {
        "var":          "generations",
        "label":        "Generation",
        "source":       "agea",
        "missing_codes": "999=Not available / No answer",
    },
    {
        "var":          "mnactic_clean",
        "label":        "Main Activity",
        "source":       "mnactic (fallback: mainact)",
        "missing_codes": "Values outside {1–9}; 77=Refusal, 88=Don't know, 99=No answer",
    },
    {
        "var":          "isco08_major_group",
        "label":        "Occupation (ISCO-08)",
        "source":       "iscoco / isco08 / isco08c / occb (first available)",
        "missing_codes": "9997=Refusal, 9998=Don't know, 9999=No answer; NaN if not in labour force / no occupation coded",
    },
    {
        "var":          "religion_clean",
        "label":        "Religious Denomination",
        "source":       "rlgblg + rlgdnm",
        "missing_codes": "rlgblg in {7=Refusal, 8=Don't know, 9=No answer}; or rlgblg=1 (religious) and rlgdnm in {55=Other, 66=Not applicable, 77=Refusal, 88=Don't know, 99=No answer}. NOTE: rlgblg=2 (no religion) → coded 0, NOT missing.",
    },
    {
        "var":          "religiosity_clean",
        "label":        "Religiosity Level",
        "source":       "rlgdgr",
        "missing_codes": "77=Refusal, 88=Don't know, 99=No answer; or value outside 0–10",
    },
    {
        "var":          "political_interest_clean",
        "label":        "Political Interest",
        "source":       "polintr",
        "missing_codes": "Values outside {1–4}; 7=Refusal, 8=Don't know, 9=No answer",
    },
    {
        "var":          "domicil_clean",
        "label":        "Domicile Type",
        "source":       "domicil",
        "missing_codes": "Values outside {1–5}; 7=Refusal, 8=Don't know, 9=No answer",
    },
    {
        "var":          "internet_time_bracket",
        "label":        "Internet Time per Day",
        "source":       "netustm",
        "missing_codes": "6666=Not applicable, 7777=Refusal, 8888=Don't know, 9999=No answer; or value < 0",
    },
]


def table_missing_data(
    db_path: Path,
    output_dir: Path,
) -> None:
    """
    Compute missing-data statistics for each of the 15 master-plot
    sociodemographic variables and save a CSV + LaTeX table.

    'Missing' means the value is recoded to NaN by recode_demographics
    (refused / don't know / out-of-range / not applicable).
    For religion_clean, respondents who said 'No religion' (rlgblg=2 → 0.0)
    are NOT counted as missing.
    """
    from .bootstrap import recode_demographics

    # Load required ESS raw columns
    _required = [
        "new_id", "agea", "edulvlb", "nobingnd", "feethngr",
        "hinctnta", "rlgdgr", "netustm", "domicil", "polintr",
    ]
    _optional = [
        "rlgdnm", "rlgblg", "mnactic", "mainact",
        "iscoco", "isco08", "isco08c", "occb",
        "brncntr", "facntr", "mocntr", "cntbrthd", "fbrncntc", "mbrncntc",
        "hincfel", "fnsdfml",
    ]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(ess11)")
    ess_avail = {r[1] for r in cursor.fetchall()}
    cursor.execute("PRAGMA table_info(demographics)")
    demo_avail = {r[1] for r in cursor.fetchall()}

    load_cols = [c for c in _required if c in ess_avail]
    load_cols += [c for c in _optional if c in ess_avail]

    df = pd.read_sql_query(
        f"SELECT {', '.join(load_cols)} FROM ess11", conn
    )
    # Pull pre-computed columns from demographics if available
    for extra_col in ("domicil_clean", "rlgdnm_clean"):
        if extra_col in demo_avail:
            df_extra = pd.read_sql_query(
                f"SELECT new_id, {extra_col} FROM demographics", conn
            )
            df = df.merge(df_extra, on="new_id", how="left")
    conn.close()

    df_rec = recode_demographics(df)
    n_total = len(df_rec)

    rows = []
    for meta in _MASTER_VAR_META:
        var = meta["var"]
        if var in df_rec.columns:
            n_missing = int(df_rec[var].isna().sum())
        else:
            # Column not derivable (source ESS column absent from DB)
            n_missing = None
        pct = (n_missing / n_total * 100) if (n_missing is not None and n_total > 0) else None
        rows.append({
            "Variable":               meta["label"],
            "Internal Name":          var,
            "ESS Source Variable(s)": meta["source"],
            "Missing ESS Codes":      meta["missing_codes"],
            "N Total":                n_total,
            "N Missing":              n_missing if n_missing is not None else "N/A (source absent)",
            "% Missing":              round(pct, 2) if pct is not None else "N/A",
        })

    df_out = pd.DataFrame(rows)
    out_csv = output_dir / "missing_data_per_sociodemographic.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # ── LaTeX table (Variable | N Missing | % Missing) ─────────────────────────
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{{Missing data per sociodemographic variable "
        r"(ESS11, $N = {:,}$). \textit{{Note}}: ``No religion'' "
        r"(\texttt{{rlgblg}} $= 2$) is recoded as 0 and is \textbf{{not}} "
        r"counted as missing.}}".format(n_total)
    )
    lines.append(r"\label{tab:missing_data_sociodem}")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Variable & $N$ Missing & \% Missing \\")
    lines.append(r"\midrule")
    for r in rows:
        n_miss  = r["N Missing"]
        pct     = r["% Missing"]
        n_str   = f"{n_miss:,}" if isinstance(n_miss, int) else str(n_miss)
        pct_str = f"{pct:.1f}" if isinstance(pct, float) else str(pct)
        var_tex = r["Variable"].replace("&", r"\&").replace("%", r"\%")
        lines.append(f"{var_tex} & {n_str} & {pct_str}\\% \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_tex = output_dir / "missing_data_per_sociodemographic.tex"
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: {out_tex}")


_ESS_COUNTRY_NAMES: dict[str, str] = {
    "AT": "Austria",        "BE": "Belgium",       "BG": "Bulgaria",
    "CH": "Switzerland",    "CY": "Cyprus",         "DE": "Germany",
    "EE": "Estonia",        "ES": "Spain",          "FI": "Finland",
    "FR": "France",         "GB": "United Kingdom", "GR": "Greece",
    "HR": "Croatia",        "HU": "Hungary",        "IE": "Ireland",
    "IL": "Israel",         "IS": "Iceland",        "IT": "Italy",
    "LT": "Lithuania",      "LV": "Latvia",         "ME": "Montenegro",
    "NL": "Netherlands",    "NO": "Norway",         "PL": "Poland",
    "PT": "Portugal",       "RS": "Serbia",         "SE": "Sweden",
    "SI": "Slovenia",       "SK": "Slovakia",       "UA": "Ukraine",
}


def table_country_respondents(
    db_path: Path,
    output_dir: Path,
) -> None:
    """
    Write a LaTeX table of respondents per country (ESS11), split into
    two side-by-side column pairs (Country | N) to keep the table compact.
    """
    conn = sqlite3.connect(db_path)
    rows_db = conn.execute(
        "SELECT cntry, COUNT(*) AS n FROM ess11 GROUP BY cntry ORDER BY cntry"
    ).fetchall()
    conn.close()

    entries = [
        (_ESS_COUNTRY_NAMES.get(code, code), n)
        for code, n in rows_db
    ]
    total_n = sum(n for _, n in entries)
    half = (len(entries) + 1) // 2          # ceiling split
    left  = entries[:half]
    right = entries[half:]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{{Respondents per country (ESS11, $N = {:,}$).}}".format(total_n)
    )
    lines.append(r"\label{tab:respondents_by_country}")
    lines.append(r"\begin{tabular}{lr|lr}")
    lines.append(r"\toprule")
    lines.append(r"Country & $N$ & Country & $N$ \\")
    lines.append(r"\midrule")

    for i in range(half):
        l_name, l_n = left[i]
        if i < len(right):
            r_name, r_n = right[i]
            lines.append(f"{l_name} & {l_n:,} & {r_name} & {r_n:,} \\\\")
        else:
            lines.append(f"{l_name} & {l_n:,} &  &  \\\\")

    lines.append(r"\midrule")
    lines.append(r"\textbf{{Total}} & \textbf{{{:,}}} &  &  \\".format(total_n))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_tex = output_dir / "respondents_by_country.tex"
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: {out_tex}")


# ── Plot: Average answer per question (population vs models) ──────────────────

def plot_avg_answer_per_question(
    db_path: Path,
    majority_votes_csvs: list,
    questions_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Scatter + line chart: mean normalised (0–1) answer per question for
    the ESS11 population and each model.

    For each question the scale is flipped if the population mean < 0.5,
    so the population is always shown at or above the mid-line.  Questions
    are ordered by topic then alphabetically within topic.

    Colours and markers match the all-models comparison plot:
    sorted model-short-name order, cycling COLOR_PALETTE / standard markers.
    """
    import ast

    _MARKERS_LOCAL = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

    # ── 1. ESS population: mean valid answer per question (normalised 0–1) ────
    df_ess = _load_ess_question_cols(db_path, questions_df)
    q_meta = questions_df.set_index("var_name")

    ess_means: dict[str, float] = {}
    for var in q_meta.index:
        if var not in df_ess.columns:
            continue
        lo = float(q_meta.loc[var, "min_scale"])
        hi = float(q_meta.loc[var, "max_scale"])
        if hi == lo:
            continue
        col = pd.to_numeric(df_ess[var], errors="coerce")
        valid = col[(col >= lo) & (col <= hi)]
        if len(valid) == 0:
            continue
        ess_means[var] = float((valid.mean() - lo) / (hi - lo))

    # ── 2. Model means: average of all seed answers per question (normalised) ─
    dfs_mv = []
    for p in majority_votes_csvs:
        if Path(p).exists():
            dfs_mv.append(pd.read_csv(p))
    if not dfs_mv:
        print("  Skipping avg-answer plot — no majority-votes CSV found.")
        return
    df_mv = pd.concat(dfs_mv, ignore_index=True)
    df_mv = df_mv[df_mv["model"].isin(MODEL_REGISTRY)].copy()
    df_mv["model_short"] = df_mv["model"].map(MODEL_REGISTRY)

    def _model_norm_majority(row) -> float:
        lo = float(row["min_scale"])
        hi = float(row["max_scale"])
        if hi == lo:
            return np.nan
        try:
            mv = float(row["majority_vote_answer"])
            if lo <= mv <= hi:
                return (mv - lo) / (hi - lo)
        except Exception:
            pass
        return np.nan

    df_mv["norm_mean"] = df_mv.apply(_model_norm_majority, axis=1)

    # ── 3. Collapse grouped questions ─────────────────────────────────────────
    grouped_vars = set(QUESTION_GROUP_MAP.keys())
    seen_groups: set[str] = set()
    entries: list[dict] = []
    for _, row in questions_df.iterrows():
        var   = str(row["var_name"])
        topic = str(row["topic"]).strip()
        if var in grouped_vars:
            grp = QUESTION_GROUP_MAP[var]
            if grp in seen_groups:
                continue
            seen_groups.add(grp)
            members = [v for v, g in QUESTION_GROUP_MAP.items() if g == grp]
            entries.append({"display_name": grp, "topic": topic, "members": members})
        else:
            entries.append({"display_name": var, "topic": topic, "members": [var]})

    # Population mean per display entry
    pop_means: dict[str, float] = {}
    for entry in entries:
        avail = [m for m in entry["members"] if m in ess_means]
        if avail:
            pop_means[entry["display_name"]] = float(np.mean([ess_means[m] for m in avail]))

    # Model mean per display entry
    model_shorts = sorted(df_mv["model_short"].dropna().unique().tolist())
    model_means: dict[str, dict[str, float]] = {ms: {} for ms in model_shorts}
    for entry in entries:
        dname = entry["display_name"]
        for ms in model_shorts:
            sub = df_mv[
                (df_mv["model_short"] == ms) &
                (df_mv["question_var"].isin(entry["members"]))
            ]
            vals = sub["norm_mean"].dropna().tolist()
            model_means[ms][dname] = float(np.mean(vals)) if vals else np.nan

    # ── 4. Order: by topic then display_name (only entries with pop data) ─────
    df_entries = pd.DataFrame([
        {"display_name": e["display_name"], "topic": e["topic"]}
        for e in entries
        if e["display_name"] in pop_means
    ]).sort_values(["topic", "display_name"]).reset_index(drop=True)

    q_names  = df_entries["display_name"].tolist()
    q_topics = df_entries["topic"].tolist()

    # ── 5. Per-question flip so population mean ≥ 0.5 ────────────────────────
    flip: dict[str, bool] = {
        q: pop_means.get(q, 0.5) < 0.5
        for q in q_names
    }

    def _f(val: float, q: str) -> float:
        if np.isnan(val):
            return np.nan
        return 1.0 - val if flip.get(q, False) else val

    pop_y   = [_f(pop_means[q], q) for q in q_names]
    model_y = {
        ms: [_f(model_means[ms].get(q, np.nan), q) for q in q_names]
        for ms in model_shorts
    }

    # ── 6. Colours / markers (same as plot_all_models_comparison) ─────────────
    all_models_sorted = sorted(model_shorts)
    model_colors  = {m: COLOR_PALETTE[i % len(COLOR_PALETTE)]   for i, m in enumerate(all_models_sorted)}
    model_markers = {m: _MARKERS_LOCAL[i % len(_MARKERS_LOCAL)] for i, m in enumerate(all_models_sorted)}
    topic_cols    = _topic_color_map(q_topics)

    # Per-model x-offsets so dots at the same question don't stack on top
    n_models = len(model_shorts)
    jitter_range = 0.80
    if n_models == 1:
        x_offsets = {model_shorts[0]: 0.0}
    else:
        offsets = np.linspace(-jitter_range / 2, jitter_range / 2, n_models)
        x_offsets = {m: float(offsets[i]) for i, m in enumerate(all_models_sorted)}

    # ── 7. Draw ───────────────────────────────────────────────────────────────
    n_q   = len(q_names)
    fig_w = max(28, n_q * 0.90)
    fig, ax = plt.subplots(figsize=(fig_w, 11))
    x = np.arange(n_q)

    # Background bands + topic labels above x-axis
    topic_spans: list[tuple[int, int, str]] = []
    cur_t: str | None = None
    s_i = 0
    for i, t in enumerate(q_topics):
        if t != cur_t:
            if cur_t is not None:
                topic_spans.append((s_i, i - 1, cur_t))
            cur_t, s_i = t, i
    if cur_t is not None:
        topic_spans.append((s_i, n_q - 1, cur_t))

    for s_i, e_i, topic in topic_spans:
        ax.axvspan(s_i - 0.5, e_i + 0.5,
                   facecolor=topic_cols[topic], alpha=0.09, zorder=0)
        ax.axvline(s_i - 0.5, color=topic_cols[topic],
                   linewidth=0.8, alpha=0.5, zorder=1)

    # Reference line at 0.5
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1.0,
               alpha=0.5, zorder=1)

    # Model dots — each model offset slightly along x so they don't overlap
    for ms in model_shorts:
        yv  = model_y[ms]
        col = model_colors[ms]
        mk  = model_markers[ms]
        xv  = x + x_offsets[ms]
        ax.scatter(xv, yv, color=col, s=170, zorder=3, marker=mk,
                   edgecolors="white", linewidths=1.2)

    # Population stars at exact question positions (drawn on top)
    ax.scatter(x, pop_y, color="black", s=520, zorder=5, marker='*',
               edgecolors="white", linewidths=1.0)

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(q_names, rotation=90, fontsize=12, ha="center")
    ax.set_xlim(-0.6, n_q - 0.4)
    ax.set_ylim(-0.04, 1.05)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel(
        "Mean normalised answer  (0–1,  per-question scale flipped so population ≥ 0.5)",
        fontsize=16, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--", color="gray", zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend along the top — population star first, then models in sorted order
    pop_handle = plt.Line2D(
        [0], [0], marker='*', color='black', linestyle='none',
        markersize=18, label="ESS11 Population",
    )
    model_handles = [
        plt.Line2D(
            [0], [0], marker=model_markers[ms], color=model_colors[ms],
            linestyle='none', markersize=12, label=ms,
        )
        for ms in model_shorts
    ]
    n_legend_cols = len(model_shorts) + 1   # one column per entity
    fig.legend(
        handles=[pop_handle] + model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=n_legend_cols,
        fontsize=14,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])   # leave room for the top legend
    out_png = output_dir / "avg_answer_per_question.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png} (+ .pdf)")

    # ── Tall version: questions on y-axis, answer on x-axis ──────────────────
    fig_h = max(28, n_q * 0.65)
    fig2, ax2 = plt.subplots(figsize=(12, fig_h))

    y = np.arange(n_q)

    # Per-model y-offsets (same spread as x-offsets in wide version)
    y_jitter_range = 1.50
    if n_models == 1:
        y_offsets = {model_shorts[0]: 0.0}
    else:
        y_offs = np.linspace(-y_jitter_range / 2, y_jitter_range / 2, n_models)
        y_offsets = {m: float(y_offs[i]) for i, m in enumerate(all_models_sorted)}

    # Alternating white / light-gray row bands (underlay)
    for qi in range(n_q):
        row_color = "#f4f4f4" if qi % 2 == 0 else "white"
        ax2.axhspan(qi - 0.5, qi + 0.5, facecolor=row_color, alpha=1.0, zorder=0)

    # Topic background bands (horizontal, on top of row alternation)
    for s_i, e_i, topic in topic_spans:
        ax2.axhspan(s_i - 0.5, e_i + 0.5,
                    facecolor=topic_cols[topic], alpha=0.09, zorder=0)
        ax2.axhline(s_i - 0.5, color=topic_cols[topic],
                    linewidth=0.8, alpha=0.5, zorder=1)

    # Reference line at 0.5
    ax2.axvline(0.5, color="#888888", linestyle="--", linewidth=1.0,
                alpha=0.5, zorder=1)

    # Model dots with y-offsets
    for ms in model_shorts:
        xv  = model_y[ms]
        col = model_colors[ms]
        mk  = model_markers[ms]
        yv  = y + y_offsets[ms]
        ax2.scatter(xv, yv, color=col, s=120, zorder=3, marker=mk,
                    edgecolors="white", linewidths=1.0)

    # Population stars at exact y positions
    ax2.scatter(pop_y, y, color="black", s=380, zorder=5, marker='*',
                edgecolors="white", linewidths=1.0)

    ax2.set_yticks(y)
    ax2.set_yticklabels(q_names, fontsize=16)
    ax2.invert_yaxis()
    ax2.set_ylim(n_q - 0.5, -0.5)   # tight: no empty space at top or bottom
    ax2.set_xlim(-0.04, 1.05)
    ax2.tick_params(axis="x", labelsize=16)
    ax2.set_xlabel(
        "Mean normalised answer  (0–1,  per-question scale flipped so population ≥ 0.5)",
        fontsize=16, fontweight="bold",
    )
    ax2.grid(axis="x", alpha=0.3, linestyle="--", color="gray", zorder=1)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Topic labels on the right side, centred within each band
    for s_i, e_i, topic in topic_spans:
        mid_y = (s_i + e_i) / 2.0
        ax2.text(
            1.01, mid_y, topic,
            va="center", ha="left", fontsize=15, fontweight="bold",
            color=topic_cols[topic],
            transform=ax2.get_yaxis_transform(), clip_on=False,
        )

    # Legend just below the axes — two rows
    n_legend_cols2 = max(1, (n_models + 1 + 1) // 2)   # ceil((n_models+1)/2)
    ax2.legend(
        handles=[pop_handle] + model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=n_legend_cols2,
        fontsize=14,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    fig2.tight_layout(rect=[0, 0, 0.88, 1.0])   # right margin for topic labels
    out_png_tall = output_dir / "avg_answer_per_question_tall.png"
    fig2.savefig(out_png_tall, dpi=150, bbox_inches="tight")
    fig2.savefig(out_png_tall.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out_png_tall} (+ .pdf)")

    # CSV
    out_csv = output_dir / "avg_answer_per_question.csv"
    df_out = pd.DataFrame({
        "question":   q_names,
        "topic":      q_topics,
        "flipped":    [flip.get(q, False) for q in q_names],
        "population": pop_y,
    })
    for ms in model_shorts:
        df_out[ms] = model_y[ms]
    df_out.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")


# ── Main entry point ───────────────────────────────────────────────────────────

def run_data_plots(
    db_path: Path,
    refusal_stats_csv: Path,
    questions_csv: Path,
    output_dir: Path,
    majority_votes_csvs: list | None = None,
) -> None:
    """
    Generate all data description plots.

    Parameters
    ----------
    db_path              : Path to ESS11.db
    refusal_stats_csv    : Path to model_refusal_stats.csv (Stage 0 output)
    questions_csv        : Path to value_laden_Qs.csv
    output_dir           : Directory to write plots into
    majority_votes_csvs  : Path(s) to model_majority_votes.csv; if None,
                           falls back to DEFAULT_MAJORITY_VOTES_CSVS from config
    """
    from .config import DEFAULT_MAJORITY_VOTES_CSVS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if majority_votes_csvs is None:
        majority_votes_csvs = DEFAULT_MAJORITY_VOTES_CSVS

    print(f"\n  Loading question metadata …")
    questions_df = _load_all_questions(questions_csv)
    print(f"  {len(questions_df)} questions in 'all' set")

    print("\n  [1] People non-answer rate per question …")
    plot_people_nonanswer(db_path, questions_df, output_dir)

    print("\n  [2] People non-answer rate per topic …")
    plot_people_nonanswer_per_topic(db_path, questions_df, output_dir)

    print("\n  [1+2] Combined: per-question with per-topic inset …")
    plot_people_nonanswer_combined(db_path, questions_df, output_dir)

    print("\n  [3] Model refusal rate per question …")
    plot_model_refusal_per_question(refusal_stats_csv, questions_df, output_dir)

    print("\n  [4] Model response heatmap …")
    plot_model_response_heatmap(refusal_stats_csv, questions_df, output_dir)

    print("\n  [5] People questions-answered histogram …")
    plot_people_questions_answered(db_path, questions_df, output_dir)

    print("\n  [6] Combined method-comp R² by LLM (all vs just_values) …")
    plot_method_comp_r2_by_llm_combined(output_dir.parent, output_dir)

    print("\n  [7] Average normalised answer per question (population vs. models) …")
    plot_avg_answer_per_question(db_path, majority_votes_csvs, questions_df, output_dir)

    print("\n  [8] Missing data table for master-plot sociodemographic variables …")
    table_missing_data(db_path, output_dir)

    print("\n  [9] Respondents by country table …")
    table_country_respondents(db_path, output_dir)

    print(f"\n  All data plots saved to: {output_dir}")
