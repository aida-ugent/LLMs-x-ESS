"""
extremeness_plots.py – Alignment-vs-response-extremeness scatter grid.

One subplot per model in a 4 × 3 grid.  Each subplot shows:
  • Individual ESS respondents as small semi-transparent dots.
  • Country-level averages as larger coloured markers (one colour per country).

The remaining empty cells at the bottom-right are merged into a single panel
that holds the country-colour legend.

Entry point
-----------
    run_extremeness_plots(db_path, questions_csv, output_dir,
                          question_set="all_answered")
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from .config import MODEL_REGISTRY, QUESTION_GROUP_MAP, DEFAULT_QUESTIONS_CSV
from .data import load_questions, q_set_key as _q_set_key

# ── Constants ──────────────────────────────────────────────────────────────────

NCOLS = 3
NROWS = 4

_COUNTRY_NAMES: dict[str, str] = {
    "AT": "Austria",       "BE": "Belgium",        "BG": "Bulgaria",
    "CH": "Switzerland",   "CY": "Cyprus",         "CZ": "Czechia",
    "DE": "Germany",       "DK": "Denmark",        "EE": "Estonia",
    "ES": "Spain",         "FI": "Finland",        "FR": "France",
    "GB": "United Kingdom","GR": "Greece",         "HR": "Croatia",
    "HU": "Hungary",       "IE": "Ireland",        "IL": "Israel",
    "IS": "Iceland",       "IT": "Italy",          "LT": "Lithuania",
    "LV": "Latvia",        "ME": "Montenegro",     "MK": "North Macedonia",
    "NL": "Netherlands",   "NO": "Norway",         "PL": "Poland",
    "PT": "Portugal",      "RO": "Romania",        "RS": "Serbia",
    "SE": "Sweden",        "SI": "Slovenia",       "SK": "Slovakia",
    "TR": "Turkey",        "UA": "Ukraine",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _country_color_map(countries: list[str]) -> dict[str, tuple]:
    """Stable colour per country using tab20 + tab20b (covers up to 40)."""
    cmap1 = cm.get_cmap("tab20")
    cmap2 = cm.get_cmap("tab20b")
    palette = [cmap1(i / 20) for i in range(20)] + [cmap2(i / 20) for i in range(20)]
    return {c: palette[i % len(palette)] for i, c in enumerate(sorted(countries))}


def _calculate_extremeness(df_ess: pd.DataFrame,
                            questions_df: pd.DataFrame) -> pd.Series:
    """
    Per-respondent mean |normalised(answer) − 0.5|, averaged across all
    valid answered questions in *questions_df*.

    Grouped questions (QUESTION_GROUP_MAP) count as one question; the first
    non-NaN variant is used per person.
    """
    valid_ranges: dict[str, tuple[float, float]] = {}
    for _, row in questions_df.iterrows():
        var = row["var_name"]
        if var in df_ess.columns:
            valid_ranges[var] = (float(row["min_scale"]), float(row["max_scale"]))

    n = len(df_ess)
    extr_sum = np.zeros(n)
    extr_cnt = np.zeros(n)
    processed_groups: set[str] = set()
    processed_vars:  set[str] = set()

    for var, (vmin, vmax) in valid_ranges.items():
        if vmax == vmin:
            continue

        if var in QUESTION_GROUP_MAP:
            group_name = QUESTION_GROUP_MAP[var]
            if group_name in processed_groups:
                continue
            group_vars = [
                v for v, g in QUESTION_GROUP_MAP.items()
                if g == group_name and v in valid_ranges
            ]
            group_extr = np.full(n, np.nan)
            for gvar in group_vars:
                gmin, gmax = valid_ranges[gvar]
                if gmax == gmin:
                    continue
                vals = pd.to_numeric(df_ess[gvar], errors="coerce").to_numpy(dtype=float)
                norm = (vals - gmin) / (gmax - gmin)
                extr = np.abs(norm - 0.5)
                extr[~((norm >= 0) & (norm <= 1))] = np.nan
                fill = np.isnan(group_extr) & np.isfinite(extr)
                group_extr[fill] = extr[fill]
            valid = np.isfinite(group_extr)
            extr_sum[valid] += group_extr[valid]
            extr_cnt[valid] += 1
            processed_groups.add(group_name)

        elif var not in processed_vars:
            vals = pd.to_numeric(df_ess[var], errors="coerce").to_numpy(dtype=float)
            valid = (vals >= vmin) & (vals <= vmax)
            norm = (vals - vmin) / (vmax - vmin)
            extr = np.abs(norm - 0.5)
            extr_sum[valid] += extr[valid]
            extr_cnt[valid] += 1
            processed_vars.add(var)

    # np.where evaluates both branches, so the zero-count rows still divide
    # 0/0 before being discarded.
    with np.errstate(invalid="ignore"):
        result = np.where(extr_cnt > 0, extr_sum / extr_cnt, np.nan)
    return pd.Series(result, index=df_ess.index, name="extremeness_score")


# ── Main entry point ───────────────────────────────────────────────────────────

def run_extremeness_plots(
    db_path: Path,
    questions_csv: Path = DEFAULT_QUESTIONS_CSV,
    output_dir: Path | None = None,
    question_set: str = "all_answered",
) -> None:
    """
    Build a 4 × 3 grid of alignment-vs-extremeness scatter plots, one per
    model.  Country averages are plotted as coloured markers; the remaining
    empty cells at the bottom are merged into a single country-legend panel.

    Parameters
    ----------
    question_set : str
        Which question set to use for computing extremeness.
        Default ``"all_answered"``.
    """
    print("\n" + "=" * 60)
    print("EXTREMENESS vs ALIGNMENT SCATTER PLOTS")
    print(f"  Question set : {question_set}")
    print("=" * 60)

    db_path   = Path(db_path)
    questions_csv = Path(questions_csv)
    if output_dir is None:
        output_dir = db_path.parent / "data_plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qset_key = _q_set_key(question_set)

    # ── Load question metadata ─────────────────────────────────────────────────
    questions_df = load_questions(question_set, questions_csv=questions_csv)

    # ── Load ESS responses ─────────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("PRAGMA table_info(ess11)")
    ess_cols = {r[1] for r in cur.fetchall()}
    conn.close()

    q_vars_present = [
        v for v in questions_df["var_name"].tolist() if v in ess_cols
    ]
    select_q = ", ".join(f'"{v}"' for v in q_vars_present)

    conn = sqlite3.connect(db_path)
    df_ess = pd.read_sql_query(
        f'SELECT new_id, cntry, {select_q} FROM ess11', conn
    )
    conn.close()
    print(f"  ESS data: {len(df_ess):,} respondents, "
          f"{len(q_vars_present)} question columns")

    # ── Compute extremeness ────────────────────────────────────────────────────
    qs_present = questions_df[questions_df["var_name"].isin(q_vars_present)].copy()
    df_ess["extremeness_score"] = _calculate_extremeness(df_ess, qs_present)
    n_valid = int(df_ess["extremeness_score"].notna().sum())
    mean_e  = float(df_ess["extremeness_score"].mean())
    print(f"  Extremeness: {n_valid:,} valid respondents  (mean={mean_e:.4f})")

    # ── Load alignment scores ──────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("PRAGMA table_info(demographics)")
    demo_cols = {r[1] for r in cur.fetchall()}
    conn.close()

    # Preserve MODEL_REGISTRY insertion order
    model_short_list = [
        ms for ms in MODEL_REGISTRY.values()
        if f"{ms}_{qset_key}_alignment" in demo_cols
    ]
    if not model_short_list:
        print(f"  WARNING: no alignment columns found for '{qset_key}' — skipping.")
        return

    align_cols = {ms: f"{ms}_{qset_key}_alignment" for ms in model_short_list}
    quoted = ", ".join(f'"{col}"' for col in align_cols.values())

    conn = sqlite3.connect(db_path)
    df_align = pd.read_sql_query(
        f'SELECT new_id, {quoted} FROM demographics', conn
    )
    conn.close()

    # ── Merge ──────────────────────────────────────────────────────────────────
    df = (
        df_ess[["new_id", "cntry", "extremeness_score"]]
        .merge(df_align, on="new_id", how="inner")
    )
    df = df.dropna(subset=["extremeness_score", "cntry"])
    print(f"  Merged: {len(df):,} respondents  ×  {len(model_short_list)} models")

    # ── Country colours ────────────────────────────────────────────────────────
    countries     = sorted(df["cntry"].dropna().unique().tolist())
    country_color = _country_color_map(countries)

    # ── Shared axis limits ─────────────────────────────────────────────────────
    all_a = np.concatenate([
        df[col].dropna().values for col in align_cols.values()
    ])
    all_e = df["extremeness_score"].dropna().values
    rng_a = np.ptp(all_a); pad_a = max(0.01, (rng_a if rng_a > 0 else 0.1) * 0.04)
    rng_e = np.ptp(all_e); pad_e = max(0.003, (rng_e if rng_e > 0 else 0.1) * 0.04)
    xlim = (float(all_a.min()) - pad_a, float(all_a.max()) + pad_a)
    ylim = (float(all_e.min()) - pad_e, float(all_e.max()) + pad_e)

    # ── Figure / GridSpec ──────────────────────────────────────────────────────
    n_models = len(model_short_list)
    n_cells  = NROWS * NCOLS  # 12

    cell_w, cell_h = 5.8, 4.8
    fig = plt.figure(figsize=(NCOLS * cell_w, NROWS * cell_h))
    gs  = GridSpec(
        NROWS, NCOLS, figure=fig,
        hspace=0.18, wspace=0.14,
        left=0.07, right=0.98, top=0.97, bottom=0.05,
    )

    # ── Model subplots ─────────────────────────────────────────────────────────
    for cell_idx, ms in enumerate(model_short_list):
        row_i = cell_idx // NCOLS
        col_i = cell_idx % NCOLS
        ax    = fig.add_subplot(gs[row_i, col_i])
        col   = align_cols[ms]

        df_m = df[["cntry", "extremeness_score", col]].dropna()
        if df_m.empty:
            ax.text(0.5, 0.5, f"{ms}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            _style_ax(ax)
            ax.text(0.5, 0.97, ms, transform=ax.transAxes,
                    ha="center", va="top", fontsize=15, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
            _edge_labels(ax, cell_idx, NCOLS, NROWS, n_models)
            continue

        # Individual respondents (sampled)
        sample_n  = min(12_000, len(df_m))
        df_sample = df_m.sample(n=sample_n, random_state=42)
        ax.scatter(
            df_sample[col], df_sample["extremeness_score"],
            alpha=0.07, s=5, color="#3A7EBF", edgecolors="none",
            rasterized=True, zorder=1,
        )

        # Country averages
        c_avg = (
            df_m.groupby("cntry")
            .agg(align=(col, "mean"), extr=("extremeness_score", "mean"))
            .reset_index()
        )
        for _, crow in c_avg.iterrows():
            ax.scatter(
                crow["align"], crow["extr"],
                color=country_color.get(crow["cntry"], "#555555"),
                s=75, zorder=3, edgecolors="white", linewidths=0.7,
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.text(0.5, 0.97, ms, transform=ax.transAxes,
                ha="center", va="top", fontsize=15, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
        _style_ax(ax)
        _edge_labels(ax, cell_idx, NCOLS, NROWS, n_models)

    # ── Legend subplot in remaining cells ─────────────────────────────────────
    if n_models < n_cells:
        last_row = (n_models - 1) // NCOLS
        last_col = (n_models - 1) % NCOLS

        if last_col < NCOLS - 1:
            # Remaining cols in last row (+ any rows below, rectangular span)
            leg_ax = fig.add_subplot(gs[last_row:, last_col + 1:])
        elif last_row < NROWS - 1:
            # Entire rows below last filled row
            leg_ax = fig.add_subplot(gs[last_row + 1:, :])
        else:
            leg_ax = None

        if leg_ax is not None:
            leg_ax.axis("off")
            handles = [
                mpatches.Patch(
                    color=country_color[c],
                    label=_COUNTRY_NAMES.get(c, c),
                )
                for c in countries
            ]
            # Estimate available columns: width of legend area in cells
            if last_col < NCOLS - 1:
                n_leg_cols = NCOLS - last_col - 1
            else:
                n_leg_cols = NCOLS
            # With ~35 countries, fit in available columns (2-3 per column)
            ncol = max(1, min(n_leg_cols * 2, len(handles)))

            leg_ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.0),
                ncol=ncol,
                fontsize=13,
                frameon=True,
                framealpha=0.6,
                edgecolor="#CCCCCC",
                title="Country averages",
                title_fontsize=15,
                handlelength=1.1,
                handleheight=1.0,
                borderpad=0.6,
                labelspacing=0.45,
                columnspacing=0.9,
            )

    # ── Super-labels ──────────────────────────────────────────────────────────
    out_path = output_dir / f"extremeness_vs_alignment_{qset_key}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Private helpers ────────────────────────────────────────────────────────────

def _style_ax(ax) -> None:
    ax.grid(True, alpha=0.22, linestyle="--", color="gray")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.spines["left"].set_color("#BBBBBB")
    ax.tick_params(labelsize=13)


def _edge_labels(ax, cell_idx: int, ncols: int, nrows: int, n_models: int) -> None:
    """Add x/y axis labels on every subplot."""
    ax.set_xlabel("Alignment", fontsize=14)
    ax.set_ylabel("Extremeness", fontsize=14)
