"""
Figures for the alignment analysis.  All plots read the bootstrap CSVs written
by `pipeline.bootstrap` (and, for the heatmaps, the per-question DB table) and
save PNGs; no GUI backend is used.

Alignment overviews
-------------------
plot_master_overview        One figure per model: all SD variables stacked vertically with CIs.
plot_countries              One figure per model: all countries sorted by alignment.
plot_all_models_comparison  All models overlaid — one figure for SD groups, one for countries.
plot_overall_alignment_comparison
                            Overall alignment per model compared across bootstrap sources.

Heatmaps
--------
plot_heatmap_by_topic       Per model: SD groups or countries (Y) × questions by topic (X).
plot_heatmap_all_models     Per model: alignment minus country mean, subgroups × countries.
plot_heatmap_cross_model_mean
                            Mean across models of the deviation from the country mean.
plot_heatmap_cross_model_overall
                            Mean across models of the deviation from the model overall mean.
plot_heatmap_cross_model_combined
                            The previous two heatmaps side by side on a shared colour scale.

Model agreement
---------------
plot_model_correlation_heatmap          Pairwise correlation over majority-vote answers.
plot_model_correlation_heatmap_person   Pairwise correlation over person-level alignment.
plot_model_correlation_heatmap_combined Both of the above in one triangular heatmap.

Deviation
---------
plot_cross_model_mean_deviation
                            Cross-model average deviation from the overall mean per
                            SD group and country, in three variants.
"""

import sqlite3
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    GROUP_LABELS, GROUP_ORDER, VARIABLE_LABELS, COLOR_PALETTE,
)
from .bootstrap import recode_demographics


# ─────────────────────────────────────────────────────────────────────────────
# Okabe-Ito palette + colour assignment
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = COLOR_PALETTE

_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']


def _model_colors(model_shorts: list[str]) -> dict[str, str]:
    """Assign colours by sorted model index, so a model keeps its colour
    across every figure regardless of which models are present."""
    return {ms: _PALETTE[i % len(_PALETTE)] for i, ms in enumerate(sorted(model_shorts))}


def _fmt_group(group, variable: str) -> str:
    """Human-readable label for a group value."""
    mapping = GROUP_LABELS.get(variable, {})
    return str(mapping.get(group, group))


def _sort_groups(groups, variable: str) -> list:
    order = GROUP_ORDER.get(variable)
    if order:
        in_order = [g for g in order if g in groups]
        rest = [g for g in groups if g not in order]
        return in_order + sorted(rest)
    try:
        return sorted(groups)
    except TypeError:
        return sorted(groups, key=str)


def _load_bootstrap_results(results_dir: Path):
    """Load all bootstrap CSV files → (df_overall | None, {var: df})."""
    overall_path = results_dir / "alignment_overall_population.csv"
    df_overall = pd.read_csv(overall_path) if overall_path.exists() else None

    var_dfs: dict[str, pd.DataFrame] = {}
    for csv_path in results_dir.glob("alignment_*.csv"):
        if csv_path.name == "alignment_overall_population.csv":
            continue
        if csv_path.stem.startswith("alignment_deviation_"):
            continue
        var_name = csv_path.stem.removeprefix("alignment_")
        var_dfs[var_name] = pd.read_csv(csv_path)

    return df_overall, var_dfs


def _load_deviation_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load alignment_deviation_*.csv files → {var: df}."""
    dev_dfs: dict[str, pd.DataFrame] = {}
    for csv_path in results_dir.glob("alignment_deviation_*.csv"):
        var_name = csv_path.stem.removeprefix("alignment_deviation_")
        dev_dfs[var_name] = pd.read_csv(csv_path)
    return dev_dfs


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Master overview – ONE figure PER MODEL, grid of CI subplots
# ─────────────────────────────────────────────────────────────────────────────

def plot_master_overview(
    bootstrap_results_dir: Path,
    output_dir: Path,
    model_short: str,
    min_n_respondents: int = 100,
    n_cols: int = 3,   # unused
    question_set_key: str = "",
) -> None:
    """
    One tall figure for *model_short*: all SD variables stacked vertically,
    colour-coded by variable, alternating background bands, variable labels
    on the right.
    Saved as master_{model_short}.png.
    """
    plt.style.use("default")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_overall, var_dfs = _load_bootstrap_results(bootstrap_results_dir)

    # One fixed colour per SD variable, so a variable reads the same in every
    # per-model figure.
    _VAR_COLORS: dict[str, str] = {
        "gender_clean":             "#E69F00",
        "ethnicmajority_clean":     "#56B4E9",
        "immigration_background":   "#0173B2",
        "education_bracket":        "#009E73",
        "income_clean":             "#CC79A7",
        "generations":              "#0072B2",
        "mnactic_clean":            "#D55E00",
        "isco08_major_group":       "#882255",
        "religion_clean":           "#44AA99",
        "religiosity_clean":        "#117733",
        "domicil_clean":            "#AA4499",
        "political_interest_clean": "#CC6677",
        "internet_time_bracket":    "#332288",
        "hincfel_clean":            "#DE8F05",
        "fnsdfml_clean":            "#CA9161",
    }

    # Overall alignment reference for this model
    overall_align: float | None = None
    if df_overall is not None:
        sub = df_overall[df_overall["model"] == model_short]
        if not sub.empty:
            overall_align = float(sub["mean_alignment"].iloc[0])

    # SD variables shown in the master figure, in display order.  cntry is
    # excluded: countries get their own figure via plot_countries.
    _MASTER_VARS = [
        "gender_clean", "ethnicmajority_clean", "immigration_background",
        "education_bracket", "income_clean", "hincfel_clean", "fnsdfml_clean",
        "generations", "mnactic_clean", "isco08_major_group", "religion_clean",
        "religiosity_clean", "political_interest_clean", "domicil_clean",
        "internet_time_bracket",
    ]
    plot_vars = [v for v in _MASTER_VARS if v in var_dfs]
    if not plot_vars:
        warnings.warn(f"No SD bootstrap data for master overview of {model_short}.")
        return

    # Build ordered (variable, group) list
    all_groups: list[tuple[str, object]] = []
    for var in plot_vars:
        df_v = var_dfs[var]
        df_v = df_v[
            (df_v["model"] == model_short) &
            (df_v["n_respondents_mean"] >= min_n_respondents)
        ]
        for g in _sort_groups(df_v["group"].unique().tolist(), var):
            all_groups.append((var, g))

    if not all_groups:
        warnings.warn(f"No data for master overview of {model_short} after N filter.")
        return

    # y_positions: 0.5-unit gap between variable sections
    y_positions: list[float] = []
    current_pos = 0.0
    current_var: str | None = None
    for var, _ in all_groups:
        if var != current_var and current_var is not None:
            current_pos += 0.5
        y_positions.append(current_pos)
        current_pos += 1.0
        current_var = var
    y_arr = np.array(y_positions)

    # Variable boundary spans for backgrounds and right-side labels
    variable_boundaries: list[tuple[int, int, str]] = []
    current_var = None
    start_idx = 0
    for idx, (var, _) in enumerate(all_groups):
        if var != current_var:
            if current_var is not None:
                variable_boundaries.append((start_idx, idx - 1, current_var))
            current_var = var
            start_idx = idx
    if current_var is not None:
        variable_boundaries.append((start_idx, len(all_groups) - 1, current_var))

    fig_height = max(12, len(all_groups) * 0.35)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Alternating background bands
    for i, (s_idx, e_idx, _) in enumerate(variable_boundaries):
        y_start = y_arr[s_idx] - 0.5
        y_end   = y_arr[e_idx] + 0.5
        if i % 2 == 0:
            ax.axhspan(y_start, y_end, facecolor="#F5F5F5", alpha=0.5, zorder=0)
        else:
            ax.axhspan(y_start, y_end, facecolor="white", alpha=1.0, zorder=0)

    # CI lines and mean dots
    for idx, (var, group) in enumerate(all_groups):
        df_v = var_dfs[var]
        sub = df_v[(df_v["model"] == model_short) & (df_v["group"] == group)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        color = _VAR_COLORS.get(var, "#2C3E50")
        ax.plot(
            [row["ci_lower"], row["ci_upper"]], [y_arr[idx], y_arr[idx]],
            color=color, linewidth=2.5, alpha=0.8, zorder=2,
        )
        ax.scatter(
            row["mean_alignment"], y_arr[idx],
            color=color, s=100, zorder=3, edgecolors="white", linewidths=1.5,
        )

    # Overall alignment reference line
    if overall_align is not None:
        ax.axvline(
            overall_align,
            color="#808080", linestyle="--", linewidth=2, alpha=0.6,
            label=f"Overall Mean: {overall_align:.3f}", zorder=1,
        )
        ax.legend(loc="lower left", fontsize=14, framealpha=0.9)

    # y-axis labels
    ax.set_yticks(y_arr)
    ax.set_yticklabels([_fmt_group(g, v) for v, g in all_groups], fontsize=18)
    ax.invert_yaxis()
    y_margin = 0.3
    ax.set_ylim(y_arr.max() + y_margin, y_arr.min() - y_margin)

    # Variable labels on right side
    for s_idx, e_idx, var in variable_boundaries:
        mid = (y_arr[s_idx] + y_arr[e_idx]) / 2
        var_label = VARIABLE_LABELS.get(var, var)
        if len(var_label) > 25:
            words = var_label.split()
            half = len(var_label) // 2
            split_at, best, acc = 0, float("inf"), 0
            for wi, w in enumerate(words):
                acc += len(w) + (1 if wi else 0)
                d = abs(acc - half)
                if d < best and acc < 30:
                    best, split_at = d, wi
            var_label = " ".join(words[:split_at + 1]) + "\n" + " ".join(words[split_at + 1:])
        ax.text(
            1.01, mid, var_label,
            va="center", ha="left", fontsize=18, fontweight="bold",
            transform=ax.get_yaxis_transform(),
            color=_VAR_COLORS.get(var, "#2C3E50"),
        )

    # x-axis limits: tight around the plotted points for THIS model only
    flat_vals: list[float] = []
    for var, group in all_groups:
        df_v = var_dfs[var]
        sub = df_v[
            (df_v["model"] == model_short) &
            (df_v["group"] == group) &
            (df_v["n_respondents_mean"] >= min_n_respondents)
        ]
        if not sub.empty:
            row = sub.iloc[0]
            flat_vals.extend([row["ci_lower"], row["ci_upper"], row["mean_alignment"]])
    if overall_align is not None:
        flat_vals.append(overall_align)
    if flat_vals:
        data_min = np.nanmin(flat_vals)
        data_max = np.nanmax(flat_vals)
        padding  = max(0.01, (data_max - data_min) * 0.05)
        ax.set_xlim(max(0.0, data_min - padding), min(1.0, data_max + padding))

    ax.set_xlabel("Alignment Score", fontsize=20, fontweight="bold")
    ax.tick_params(axis="x", labelsize=16)
    model_title = model_short.split("/")[-1] if "/" in model_short else model_short
    qs_label = f"  [{question_set_key}]" if question_set_key else ""
    # ax.set_title(
    #     f"Alignment Across All Demographic Variables\n{model_title}{qs_label}",
    #     fontsize=22, fontweight="bold", pad=5,
    # )

    ax.grid(axis="x", alpha=0.3, linestyle="--", color="gray", zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_color("#CCCCCC")

    plt.tight_layout()
    plt.subplots_adjust(right=0.88, top=0.98)

    out_path = output_dir / f"master_alignment_all_demographics_{model_short}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Countries CI plot – ONE figure PER MODEL, all countries sorted by alignment
# ─────────────────────────────────────────────────────────────────────────────

_COUNTRY_NAMES: dict[str, str] = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland",
    "CY": "Cyprus", "CZ": "Czechia", "DE": "Germany", "DK": "Denmark",
    "EE": "Estonia", "ES": "Spain", "FI": "Finland", "FR": "France",
    "GB": "United Kingdom", "GR": "Greece", "HR": "Croatia", "HU": "Hungary",
    "IE": "Ireland", "IL": "Israel", "IS": "Iceland", "IT": "Italy",
    "LT": "Lithuania", "LV": "Latvia", "ME": "Montenegro", "MK": "North Macedonia",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RO": "Romania", "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
    "SK": "Slovakia", "TR": "Turkey", "UA": "Ukraine",
}


def plot_countries(
    bootstrap_results_dir: Path,
    output_dir: Path,
    model_short: str,
    n_top: int = 6,        # unused
    n_bottom: int = 6,     # unused
    min_n_respondents: int = 100,
    question_set_key: str = "",
) -> None:
    """
    One figure for *model_short*: ALL countries sorted by mean alignment (lowest → highest).
    Saved as countries_{model_short}.png.
    """
    plt.style.use("default")
    output_dir.mkdir(parents=True, exist_ok=True)
    cntry_csv = bootstrap_results_dir / "alignment_cntry.csv"
    if not cntry_csv.exists():
        warnings.warn("alignment_cntry.csv not found; skipping countries plot.")
        return

    df_overall_path = bootstrap_results_dir / "alignment_overall_population.csv"
    df_overall = pd.read_csv(df_overall_path) if df_overall_path.exists() else None

    df_all = pd.read_csv(cntry_csv)
    df_all = df_all[df_all["n_respondents_mean"] >= min_n_respondents]

    # Consistent colour assignment across all models
    all_models = sorted(df_all["model"].unique().tolist())
    colors = _model_colors(all_models)
    color = colors.get(model_short, _PALETTE[0])

    df = df_all[df_all["model"] == model_short].copy()
    if df.empty:
        warnings.warn(f"No country data for model {model_short}; skipping.")
        return

    # Sort all countries by mean alignment ascending
    df = df.sort_values("mean_alignment", ascending=True).reset_index(drop=True)

    overall_align: float | None = None
    if df_overall is not None:
        sub = df_overall[df_overall["model"] == model_short]
        if not sub.empty:
            overall_align = float(sub["mean_alignment"].iloc[0])

    fig_h = max(6, len(df) * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    # Alternating gray/white backgrounds per country
    for idx in range(len(df)):
        if idx % 2 == 0:
            ax.axhspan(idx - 0.5, idx + 0.5, facecolor="#F5F5F5", alpha=0.5, zorder=0)

    for idx, (_, row) in enumerate(df.iterrows()):
        ax.plot(
            [row["ci_lower"], row["ci_upper"]], [idx, idx],
            color=color, linewidth=2, alpha=0.7, zorder=1,
        )
        ax.scatter(
            row["mean_alignment"], idx,
            color=color, s=100, zorder=2, edgecolors="white", linewidths=1.5,
        )
        ax.text(
            1.02, idx,
            f"N={int(row['n_respondents_mean'])}",
            va="center", ha="left", fontsize=11, color="gray",
            transform=ax.get_yaxis_transform(),
        )

    if overall_align is not None:
        ax.axvline(
            overall_align,
            color="gray", linestyle="--", linewidth=2, alpha=0.7,
            label=f"Overall Mean: {overall_align:.3f}", zorder=0,
        )
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    country_labels = [_COUNTRY_NAMES.get(c, c) for c in df["group"]]
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(country_labels, fontsize=13)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlabel("Alignment Score", fontsize=16, fontweight="bold")
    ax.set_ylabel("Country", fontsize=16, fontweight="bold")
    model_title = model_short.split("/")[-1] if "/" in model_short else model_short
    qs_label = f"  [{question_set_key}]" if question_set_key else ""
    # ax.set_title(
    #     f"Alignment by Country\n{model_title}{qs_label}",
    #     fontsize=18, fontweight="bold", pad=20,
    # )
    all_vals = df[["ci_lower", "ci_upper", "mean_alignment"]].values.flatten()
    ax.set_xlim(max(0, np.nanmin(all_vals) - 0.05), min(1, np.nanmax(all_vals) + 0.05))
    ax.grid(axis="x", alpha=0.3, linestyle="--", color="gray")
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.spines["top"].set_color("#CCCCCC")
    ax.spines["right"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_color("#CCCCCC")

    plt.tight_layout()
    out_path = output_dir / f"countries_alignment_by_country_{model_short}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2b. All-models comparison – SD groups + countries, all models overlaid
# ─────────────────────────────────────────────────────────────────────────────

_VAR_COLORS_MASTER: dict[str, str] = {
    "gender_clean":             "#E69F00",
    "ethnicmajority_clean":     "#56B4E9",
    "immigration_background":   "#0173B2",
    "education_bracket":        "#009E73",
    "income_clean":             "#CC79A7",
    "generations":              "#0072B2",
    "mnactic_clean":            "#D55E00",
    "isco08_major_group":       "#882255",
    "religion_clean":           "#44AA99",
    "religiosity_clean":        "#117733",
    "domicil_clean":            "#AA4499",
    "political_interest_clean": "#CC6677",
    "internet_time_bracket":    "#332288",
    "hincfel_clean":            "#DE8F05",
    "fnsdfml_clean":            "#CA9161",
}

_MASTER_VAR_ORDER = [
    "gender_clean", "ethnicmajority_clean", "immigration_background",
    "education_bracket", "income_clean", "hincfel_clean", "fnsdfml_clean",
    "generations", "mnactic_clean", "isco08_major_group", "religion_clean",
    "religiosity_clean", "political_interest_clean", "domicil_clean",
    "internet_time_bracket",
]


def plot_all_models_comparison(
    bootstrap_results_dir: Path,
    output_dir: Path,
    question_set_key: str = "",
    min_n_respondents: int = 100,
) -> None:
    """
    Two separate figures with ALL models overlaid using distinct colours + marker shapes,
    models jittered vertically per row so CIs don't fully overlap:
      - all_models_comparison_sd.png    - sociodemographic groups
      - all_models_comparison_cntry.png - countries
    """
    plt.style.use("default")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_overall, var_dfs = _load_bootstrap_results(bootstrap_results_dir)

    # Collect all models
    all_models: list[str] = []
    if df_overall is not None:
        all_models = sorted(df_overall["model"].unique().tolist())
    if not all_models:
        for df_v in var_dfs.values():
            all_models = sorted(set(all_models) | set(df_v["model"].unique()))
    if not all_models:
        warnings.warn("No model data found; skipping all-models comparison plot.")
        return

    n_models = len(all_models)
    model_colors  = {m: _PALETTE[i % len(_PALETTE)]  for i, m in enumerate(all_models)}
    model_markers = {m: _MARKERS[i % len(_MARKERS)] for i, m in enumerate(all_models)}

    jitter_range = 0.28
    if n_models == 1:
        jitter = {all_models[0]: 0.0}
    else:
        offsets = np.linspace(-jitter_range / 2, jitter_range / 2, n_models)
        jitter  = {m: float(offsets[i]) for i, m in enumerate(all_models)}

    # Build row lists
    sd_rows: list[tuple[str, object]] = []
    for var in _MASTER_VAR_ORDER:
        if var not in var_dfs:
            continue
        df_v = var_dfs[var]
        valid_groups: set = set()
        for m in all_models:
            sub = df_v[(df_v["model"] == m) & (df_v["n_respondents_mean"] >= min_n_respondents)]
            valid_groups |= set(sub["group"].unique())
        for g in _sort_groups(list(valid_groups), var):
            sd_rows.append((var, g))

    cntry_df = var_dfs.get("cntry")
    country_rows: list[str] = []
    if cntry_df is not None:
        cntry_filt = cntry_df[cntry_df["n_respondents_mean"] >= min_n_respondents]
        country_means = cntry_filt.groupby("group")["mean_alignment"].mean()
        country_rows  = country_means.sort_values(ascending=True).index.tolist()

    if not sd_rows and not country_rows:
        warnings.warn("No data for all-models comparison plot.")
        return

    qs_label = f"  [{question_set_key}]" if question_set_key else ""

    def _draw_comparison_fig(
        rows: list[tuple[str, object]],
        fname: str,
        title: str,
        legend_pos: str | None = None,
    ):
        if not rows:
            return

        y_pos: dict[tuple, float] = {}
        current_y = 0.0
        prev_var  = None
        for var, group in rows:
            if var != prev_var and prev_var is not None:
                current_y += 0.5
            y_pos[(var, group)] = current_y
            current_y += 1.0
            prev_var = var

        var_boundaries: list[tuple[float, float, str]] = []
        current_var: str | None = None
        first_y = last_y = 0.0
        for var, group in rows:
            y = y_pos[(var, group)]
            if var != current_var:
                if current_var is not None:
                    var_boundaries.append((first_y, last_y, current_var))
                current_var = var
                first_y = y
            last_y = y
        if current_var is not None:
            var_boundaries.append((first_y, last_y, current_var))

        n_rows_fig = len(rows)
        fig_h = max(16, n_rows_fig * 0.32)
        fig, ax = plt.subplots(figsize=(14, fig_h))

        # Alternating gray/white backgrounds
        if all(var == "cntry" for var, _ in rows):
            # Per-country alternating bands (matches IPW country plot style)
            for i, (var, group) in enumerate(rows):
                if i % 2 == 0:
                    y = y_pos[(var, group)]
                    ax.axhspan(y - 0.5, y + 0.5, facecolor="#F5F5F5", alpha=0.5, zorder=0)
        else:
            for i, (fy, ly, _var) in enumerate(var_boundaries):
                fc = "#F5F5F5" if i % 2 == 0 else "white"
                ax.axhspan(fy - 0.5, ly + 0.5, facecolor=fc, alpha=0.5, zorder=0)

        def _get_df(var):
            if var == "cntry":
                return cntry_df
            return var_dfs.get(var)

        # Dashed vertical line per model at its overall mean
        if df_overall is not None:
            for m in all_models:
                sub_ov = df_overall[df_overall["model"] == m]
                if not sub_ov.empty:
                    ov_mean = float(sub_ov.iloc[0]["mean_alignment"])
                    ax.axvline(ov_mean, color=model_colors[m], linestyle="--",
                               linewidth=1.4, alpha=0.55, zorder=1)

        for var, group in rows:
            df_v = _get_df(var)
            if df_v is None:
                continue
            y_base = y_pos[(var, group)]
            for m in all_models:
                sub = df_v[
                    (df_v["model"] == m) &
                    (df_v["group"]  == group) &
                    (df_v["n_respondents_mean"] >= min_n_respondents)
                ]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                y   = y_base + jitter[m]
                ax.plot([row["ci_lower"], row["ci_upper"]], [y, y],
                        color=model_colors[m], linewidth=1.5, alpha=0.6, zorder=2)
                ax.scatter(row["mean_alignment"], y,
                           color=model_colors[m], marker=model_markers[m],
                           s=55, zorder=3, edgecolors="white", linewidths=0.8)

        all_y_ticks:  list[float] = []
        all_y_labels: list[str]   = []
        for var, group in rows:
            all_y_ticks.append(y_pos[(var, group)])
            if var == "cntry":
                all_y_labels.append(_COUNTRY_NAMES.get(str(group), str(group)))
            else:
                all_y_labels.append(_fmt_group(group, var))

        ax.set_yticks(all_y_ticks)
        ax.set_yticklabels(all_y_labels, fontsize=15)
        ax.invert_yaxis()
        y_all = list(y_pos.values())
        ax.set_ylim(max(y_all) + 0.8, min(y_all) - 0.8)

        for fy, ly, var in var_boundaries:
            mid = (fy + ly) / 2
            if var == "cntry":
                lbl = "Country"
                color_lbl = "#2C3E50"
            else:
                lbl = VARIABLE_LABELS.get(var, var)
                color_lbl = _VAR_COLORS_MASTER.get(var, "#2C3E50")
            if len(lbl) > 22:
                words = lbl.split()
                half  = len(lbl) // 2
                split_at, best, acc = 0, float("inf"), 0
                for wi, w in enumerate(words):
                    acc += len(w) + (1 if wi else 0)
                    d = abs(acc - half)
                    if d < best and acc < 30:
                        best, split_at = d, wi
                lbl = " ".join(words[:split_at + 1]) + "\n" + " ".join(words[split_at + 1:])
            ax.text(1.01, mid, lbl, va="center", ha="left", fontsize=15, fontweight="bold",
                    transform=ax.get_yaxis_transform(), color=color_lbl)

        flat_vals: list[float] = []
        for var, group in rows:
            df_v = _get_df(var)
            if df_v is None:
                continue
            for m in all_models:
                sub = df_v[(df_v["model"] == m) & (df_v["group"] == group) &
                           (df_v["n_respondents_mean"] >= min_n_respondents)]
                if not sub.empty:
                    r = sub.iloc[0]
                    flat_vals.extend([r["ci_lower"], r["ci_upper"], r["mean_alignment"]])
        if flat_vals:
            dmin    = np.nanmin(flat_vals)
            dmax    = np.nanmax(flat_vals)
            padding = max(0.01, (dmax - dmin) * 0.05)
            ax.set_xlim(max(0.0, dmin - padding), min(1.0, dmax + padding))

        from matplotlib.lines import Line2D
        # Add dashed-line legend entries for overall means
        legend_handles = [
            Line2D([0], [0],
                   marker=model_markers[m], color=model_colors[m],
                   markerfacecolor=model_colors[m], markersize=9,
                   linewidth=1.5, label=m)
            for m in all_models
        ]
        if df_overall is not None:
            for m in all_models:
                sub_ov = df_overall[df_overall["model"] == m]
                if not sub_ov.empty:
                    legend_handles.append(
                        Line2D([0], [0], color=model_colors[m], linestyle="--",
                               linewidth=1.4, label=f"{m} (overall mean)")
                    )

        if legend_pos == "bottom":
            ax.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.07),
                fontsize=13, framealpha=0.9,
                title="Model", title_fontsize=14,
                ncol=5,
                borderaxespad=0,
            )
        # legend_pos is None → no legend

        # ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.set_xlabel("Alignment Score", fontsize=18, fontweight="bold")
        ax.tick_params(axis="x", labelsize=15)
        ax.grid(axis="x", alpha=0.3, linestyle="--", color="gray", zorder=1)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.spines["left"].set_color("#CCCCCC")

        plt.tight_layout()
        plt.subplots_adjust(right=0.78, top=0.98)
        out_path = output_dir / fname
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    _draw_comparison_fig(
        rows=sd_rows,
        fname="all_models_comparison_alignment_scores_sd.png",
        title=f"Alignment Scores — All Models Comparison (SD){qs_label}",
        legend_pos=None,
    )
    _draw_comparison_fig(
        rows=[("cntry", c) for c in country_rows],
        fname="all_models_comparison_alignment_scores_cntry.png",
        title=f"Alignment Scores — All Models Comparison (Countries){qs_label}",
        legend_pos="bottom",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SD variables shown in the topic heatmap, in row order.
_HEATMAP_TOPIC_SD_VARS = [
    "gender_clean",
    "ethnicmajority_clean",
    "immigration_background",
    "education_bracket",
    "income_clean",
    "hincfel_clean",
    "fnsdfml_clean",
    "generations",
    "mnactic_clean",
    "isco08_major_group",
    "religion_clean",
    "religiosity_clean",
    "political_interest_clean",
    "domicil_clean",
    "internet_time_bracket",
]

_HEATMAP_TOPIC_SD_LABELS = {
    "gender_clean":             "GENDER",
    "ethnicmajority_clean":     "ETHNICITY",
    "immigration_background":   "IMMIGRATION BG",
    "education_bracket":        "EDUCATION",
    "income_clean":             "INCOME",
    "hincfel_clean":            "INCOME FEELING",
    "fnsdfml_clean":            "CHILDHOOD FINANCES",
    "generations":              "GENERATION",
    "mnactic_clean":            "MAIN ACTIVITY",
    "isco08_major_group":       "OCCUPATION",
    "religion_clean":           "RELIGION",
    "religiosity_clean":        "RELIGIOSITY",
    "domicil_clean":            "DOMICILE",
    "political_interest_clean": "POLITICAL INTEREST",
    "internet_time_bracket":    "INTERNET TIME",
}


def plot_heatmap_by_topic(
    db_path,
    question_set_key: str,
    model_short: str,
    questions_df: pd.DataFrame,
    output_dir: Path,
    mode: str = "sd_groups",   # "sd_groups" | "countries"
    min_n: int = 50,
    sd_variable: str = "",   # unused
) -> None:
    """
    One heatmap per model per mode.

    mode="sd_groups": Y = all SD groups stacked (all variables with header rows),
                      X = questions sorted by topic.
    mode="countries": Y = countries, X = questions sorted by topic.

    Saved as heatmap_sdgroups_{model_short}.png or heatmap_countries_{model_short}.png.
    """
    plt.style.use("default")
    output_dir.mkdir(parents=True, exist_ok=True)

    table = f"per_question_{question_set_key}"
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"].tolist()
    if table not in tables:
        conn.close()
        warnings.warn(f"Table '{table}' not found; skipping topic heatmap for {model_short}.")
        return

    df_pq = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    # Load ESS demographics needed for recoding
    ess_base = (
        "e.new_id, e.cntry, e.agea, e.edulvlb, e.nobingnd, e.feethngr, "
        "e.hinctnta, e.rlgdgr, e.netustm, e.domicil, e.polintr, "
        "e.contplt, e.donprty, e.badge, e.sgnptit, e.pbldmna, e.bctprd, e.pstplonl, "
        "e.brncntr, e.facntr, e.mocntr"
    )
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(ess11)")
    ess_avail = {r[1] for r in cursor.fetchall()}
    for opt in ["rlgblg", "rlgdnm", "mnactic", "isco08", "iscoco",
                "hincfel", "fnsdfml", "cntbrthd", "fbrncntc", "mbrncntc"]:
        if opt in ess_avail:
            ess_base += f", e.{opt}"
    df_ess = pd.read_sql_query(f"SELECT {ess_base} FROM ess11 e", conn)
    conn.close()

    df_ess = recode_demographics(df_ess)
    df = df_pq.merge(df_ess, on="new_id", how="left")

    # Per-question columns for this model
    prefix = f"{model_short}_"
    q_cols = [c for c in df_pq.columns if c.startswith(prefix)]
    if not q_cols:
        warnings.warn(f"No per-question columns for {model_short} in {table}.")
        return
    q_vars = [c[len(prefix):] for c in q_cols]

    # Sort questions by topic then var name
    q_meta = questions_df.set_index("var_name")
    valid = [(v, c) for v, c in zip(q_vars, q_cols) if v in q_meta.index]
    if not valid:
        warnings.warn("No matching questions in metadata; skipping heatmap.")
        return
    q_vars_f, q_cols_f = zip(*valid)
    topics = [str(q_meta.loc[v, "topic"]).strip().title() for v in q_vars_f]
    order = sorted(range(len(q_vars_f)), key=lambda i: (topics[i], q_vars_f[i]))
    q_vars_f = [q_vars_f[i] for i in order]
    q_cols_f = [q_cols_f[i] for i in order]
    topics   = [topics[i]   for i in order]

    if mode == "countries":
        _save_heatmap_countries(df, q_vars_f, q_cols_f, topics, model_short, output_dir, min_n, question_set_key)
    else:
        _save_heatmap_sdgroups(df, q_vars_f, q_cols_f, topics, model_short, output_dir, min_n, question_set_key)


def _save_heatmap_countries(df, q_vars_f, q_cols_f, topics, model_short, output_dir, min_n, question_set_key=""):
    groups = sorted(df["cntry"].dropna().unique())
    if not groups:
        return
    matrix = np.full((len(groups), len(q_cols_f)), np.nan)
    for g_idx, g in enumerate(groups):
        mask = df["cntry"] == g
        for q_idx, qcol in enumerate(q_cols_f):
            vals = pd.to_numeric(df.loc[mask, qcol], errors="coerce").dropna()
            if len(vals) >= min_n:
                matrix[g_idx, q_idx] = float(vals.mean())

    # Sort countries by overall alignment score (mean across all questions), highest first
    row_means = np.nanmean(matrix, axis=1)
    sort_order = np.argsort(row_means)[::-1]
    matrix = matrix[sort_order]
    groups = [groups[i] for i in sort_order]

    _draw_heatmap(
        matrix=matrix,
        row_labels=list(groups),
        q_vars_f=q_vars_f,
        topics=topics,
        model_short=model_short,
        row_axis_label="Country (ordered by mean alignment)",
        fname=f"heatmap_mean_alignment_per_question_countries_{model_short}.png",
        output_dir=output_dir,
        row_separators=[],
        question_set_key=question_set_key,
        cmap="RdBu_r",
    )


def _save_heatmap_sdgroups(df, q_vars_f, q_cols_f, topics, model_short, output_dir, min_n, question_set_key=""):
    row_labels: list[str] = []
    row_separators: list[int] = []   # y-positions of separator lines
    matrix_rows: list[np.ndarray] = []

    for sd_var in _HEATMAP_TOPIC_SD_VARS:
        if sd_var not in df.columns:
            continue
        groups = _sort_groups(df[sd_var].dropna().unique().tolist(), sd_var)
        if not groups:
            continue

        if row_labels:   # separator before every block except the first
            row_separators.append(len(row_labels))

        block_label = _HEATMAP_TOPIC_SD_LABELS.get(sd_var, sd_var)
        for g in groups:
            mask = df[sd_var] == g
            row_vals = np.full(len(q_cols_f), np.nan)
            for q_idx, qcol in enumerate(q_cols_f):
                vals = pd.to_numeric(df.loc[mask, qcol], errors="coerce").dropna()
                if len(vals) >= min_n:
                    row_vals[q_idx] = float(vals.mean())
            matrix_rows.append(row_vals)
            row_labels.append(f"{block_label}: {_fmt_group(g, sd_var)}")

    if not matrix_rows:
        warnings.warn(f"No data for sd_groups heatmap for {model_short}.")
        return

    _draw_heatmap(
        matrix=np.vstack(matrix_rows),
        row_labels=row_labels,
        q_vars_f=q_vars_f,
        topics=topics,
        model_short=model_short,
        row_axis_label="Socio-Demographic Group",
        fname=f"heatmap_mean_alignment_per_question_sdgroups_{model_short}.png",
        output_dir=output_dir,
        row_separators=row_separators,
        question_set_key=question_set_key,
    )


def _draw_heatmap(matrix, row_labels, q_vars_f, topics, model_short,
                  row_axis_label, fname, output_dir, row_separators, question_set_key="",
                  cmap="RdYlGn"):
    """Shared routine: draws and saves one heatmap figure."""
    import matplotlib.colors as mcolors

    n_rows, n_cols = matrix.shape
    fig_w = max(14, n_cols * 0.35)
    fig_h = max(6, n_rows * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#D3D3D3")
    masked = np.ma.masked_invalid(matrix)

    valid_vals = matrix[~np.isnan(matrix)]
    if len(valid_vals):
        vmin = float(np.nanmin(valid_vals))
        vmax = float(np.nanmax(valid_vals))
    else:
        vmin, vmax = 0.4, 0.9

    # For diverging colormaps, centre the norm on the data midpoint
    if cmap in ("RdBu", "RdBu_r", "coolwarm", "bwr"):
        vcenter = (vmin + vmax) / 2
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=0.4, vmax=0.9)

    im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, norm=norm)
    plt.colorbar(im, ax=ax, label="Mean alignment score", fraction=0.025, pad=0.02)

    # X-axis: question var names, coloured by topic
    unique_topics = list(dict.fromkeys(topics))
    tab20 = plt.cm.tab20
    topic_colors = {t: tab20(i / max(len(unique_topics), 1)) for i, t in enumerate(unique_topics)}

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(q_vars_f, rotation=90, fontsize=10)
    for tick, topic in zip(ax.get_xticklabels(), topics):
        tick.set_color(topic_colors[topic])

    # White vertical lines between topics
    prev_t = None
    for i, t in enumerate(topics):
        if t != prev_t and i > 0:
            ax.axvline(i - 0.5, color="white", linewidth=2.0)
        prev_t = t

    # Y-axis
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)

    # White horizontal lines between SD variable blocks
    for sep_y in row_separators:
        ax.axhline(sep_y - 0.5, color="white", linewidth=2.5)

    # Topic legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=topic_colors[t]) for t in unique_topics]
    ax.legend(
        legend_handles, unique_topics,
        title="Topic", loc="upper left",
        bbox_to_anchor=(1.02, 1), fontsize=11, framealpha=0.9,
    )

    qs_label = f"  [{question_set_key}]" if question_set_key else ""
    # ax.set_title(
    #     f"{model_short}{qs_label} — Mean alignment score per question\n"
    #     f"Rows: {row_axis_label}  |  Colour = topic",
    #     fontsize=15, fontweight="bold",
    # )
    ax.set_xlabel("Question (colour = topic)", fontsize=13)
    ax.set_ylabel(row_axis_label, fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")




# ─────────────────────────────────────────────────────────────────────────────
# 4.  All-models heatmap: alignment diff from country mean (one per model)
# ─────────────────────────────────────────────────────────────────────────────

# Subgroup row ordering + display labels for the all-models heatmaps.
# Each entry is (SD variable, subgroup labels in the order they are drawn).
_ALL_MODELS_CAT_GROUPS = [
    ("gender_clean",           ["Male", "Female", "Non-binary"]),
    ("ethnicmajority_clean",   ["Ethnic majority", "Ethnic minority"]),
    ("immigration_background", ["No Migration Background", "Western Migration Background",
                                 "Non-Western Migration Background", "Mixed Migration Background"]),
    ("education_bracket",      ["≤ Lower secondary", "Lower secondary",
                                 "Upper secondary", "Post-secondary non-tertiary",
                                 "Short-cycle tertiary", "Bachelor", "Master", "Doctoral"]),
    ("income_cat",             [f"Income D{i}" for i in range(1, 11)]),
    ("hincfel_clean",          ["Living comfortably", "Coping", "Difficult", "Very difficult"]),
    ("fnsdfml_clean",          ["Always", "Often", "Sometimes", "Hardly ever", "Never"]),
    ("generation",             ["Gen Z (<26)", "Millennials (26-42)", "Gen X (43-58)",
                                 "Baby Boomers (59-77)", "Silent Gen (78+)"]),
    ("mnactic_clean",          ["Paid work", "Education", "Unemployed, seeking",
                                 "Unemployed, not seeking", "Permanently sick / disabled",
                                 "Retired", "Military / community service",
                                 "Housework / children", "Other"]),
    ("isco08_major_group",     ["Armed forces", "Managers", "Professionals",
                                 "Technicians / Associates", "Clerical support workers",
                                 "Service and sales workers", "Agricultural workers",
                                 "Craft and trades workers", "Plant / machine operators",
                                 "Elementary occupations"]),
    ("religion_clean",         ["No Religion", "Roman Catholic", "Protestant",
                                 "Eastern Orthodox", "Other Christian", "Jewish",
                                 "Islam", "Eastern religions", "Other non-Christian"]),
    ("religiosity_cat",        ["Religiosity: Low (0-2)", "Religiosity: Medium (3-7)",
                                 "Religiosity: High (8-10)"]),
    ("internet_time_cat",      ["Internet time: None", "Internet time: 1-30 min",
                                 "Internet time: 31-60 min", "Internet time: 61-180 min",
                                 "Internet time: 181-300 min", "Internet time: 301-420 min",
                                 "Internet time: 421+ min"]),
    ("domicile",               ["Big city", "Suburbs/outskirts", "Town/small city",
                                 "Country village", "Farm/countryside"]),
    ("political_interest_cat", ["Politics: Very interested", "Politics: Quite interested",
                                 "Politics: Hardly interested",
                                 "Politics: Not at all interested"]),
]

_ALL_MODELS_CAT_LABELS: dict[str, str] = {
    "gender_clean":           "Gender",
    "ethnicmajority_clean":   "Ethnicity",
    "immigration_background": "Immigration Background",
    "education_bracket":      "Education Level",
    "income_cat":             "Income Decile",
    "hincfel_clean":          "Income Feeling",
    "fnsdfml_clean":          "Childhood Finances",
    "generation":             "Generation",
    "mnactic_clean":          "Main Activity",
    "isco08_major_group":     "Occupation (ISCO-08)",
    "religion_clean":         "Religious Denomination",
    "religiosity_cat":        "Religiosity",
    "internet_time_cat":      "Internet Time (Daily)",
    "domicile":               "Domicile",
    "political_interest_cat": "Political Interest",
}


def _recode_all_models_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the subgroup columns the all-models heatmaps plot as rows.

    Produces the categorical labels used by _ALL_MODELS_CAT_GROUPS (generation
    cohort, education bracket, income bracket, religiosity, …) from the raw ESS
    columns, on a copy of *df*.
    """

    def recode_generation(age):
        if pd.isna(age) or age >= 999:  return None
        if age < 26:  return "Gen Z (<26)"
        if age < 43:  return "Millennials (26-42)"
        if age < 59:  return "Gen X (43-58)"
        if age < 78:  return "Baby Boomers (59-77)"
        return "Silent Gen (78+)"

    def recode_education(val):
        if pd.isna(val) or val in [5555, 7777, 8888, 9999]:  return None
        if val < 200:  return "≤ Lower secondary"
        if val < 300:  return "Lower secondary"
        if val < 400:  return "Upper secondary"
        if val < 500:  return "Post-secondary non-tertiary"
        if val < 600:  return "Short-cycle tertiary"
        if val < 700:  return "Bachelor"
        if val < 800:  return "Master"
        return "Doctoral"

    def recode_gender(val):
        return {1: "Male", 2: "Female", 3: "Non-binary"}.get(val)

    def recode_ethnic(val):
        return {1: "Ethnic majority", 2: "Ethnic minority"}.get(val)
    def recode_hincfel(val):
        return {1: "Living comfortably", 2: "Coping",
                3: "Difficult", 4: "Very difficult"}.get(int(val)) if pd.notna(val) else None

    def recode_fnsdfml(val):
        return {1: "Always", 2: "Often", 3: "Sometimes",
                4: "Hardly ever", 5: "Never"}.get(int(val)) if pd.notna(val) else None

    def recode_religion(row):
        denom_map = {
            1: "Roman Catholic", 2: "Protestant", 3: "Eastern Orthodox",
            4: "Other Christian", 5: "Jewish", 6: "Islam",
            7: "Eastern religions", 8: "Other non-Christian",
        }
        blg = row.get("rlgblg")
        dnm = row.get("rlgdnm")
        if pd.notna(blg):
            if blg == 2:
                return "No Religion"
            if blg == 1 and pd.notna(dnm) and dnm not in [55, 66, 77, 88, 99]:
                return denom_map.get(int(dnm))
        elif pd.notna(dnm) and dnm not in [55, 66, 77, 88, 99]:
            return denom_map.get(int(dnm))
        return None

    def recode_mnactic(val):
        return {1: "Paid work", 2: "Education", 3: "Unemployed, seeking",
                4: "Unemployed, not seeking", 5: "Permanently sick / disabled",
                6: "Retired", 7: "Military / community service",
                8: "Housework / children", 9: "Other"}.get(int(val)) if pd.notna(val) else None

    def recode_isco(val):
        if pd.isna(val) or val in [9997, 9998, 9999]:
            return None
        major = int(val) // 1000
        return {0: "Armed forces", 1: "Managers", 2: "Professionals",
                3: "Technicians / Associates", 4: "Clerical support workers",
                5: "Service and sales workers", 6: "Agricultural workers",
                7: "Craft and trades workers", 8: "Plant / machine operators",
                9: "Elementary occupations"}.get(major)

    _western = {
        "AT","BE","CH","CY","CZ","DE","DK","EE","ES","FI","FR","GB",
        "GR","HR","HU","IE","IL","IS","IT","LT","LU","LV","ME","MK",
        "NL","NO","PL","PT","RO","RS","SE","SI","SK","TR","UA",
        "AU","CA","NZ","US",
    }

    def recode_immigration(row):
        brn = row.get("brncntr")
        if pd.isna(brn) or brn in [7, 8, 9]:
            return None
        fac = row.get("facntr"); moc = row.get("mocntr")
        fac_v = pd.notna(fac) and fac in [1, 2]
        moc_v = pd.notna(moc) and moc in [1, 2]
        if brn == 1 and (not fac_v or fac == 1) and (not moc_v or moc == 1):
            return "No Migration Background"
        origins = []
        fbc = row.get("fbrncntc"); mbc = row.get("mbrncntc")
        if fac_v and fac == 2 and pd.notna(fbc):
            origins.append(str(fbc).upper() in _western)
        if moc_v and moc == 2 and pd.notna(mbc):
            origins.append(str(mbc).upper() in _western)
        if not origins:
            cbc = row.get("cntbrthd")
            if brn == 2 and pd.notna(cbc):
                origins.append(str(cbc).upper() in _western)
        if not origins:
            return None
        if all(origins):  return "Western Migration Background"
        if not any(origins): return "Non-Western Migration Background"
        return "Mixed Migration Background"

    def recode_income(val):
        if pd.notna(val) and 1 <= val <= 10:
            return f"Income D{int(val)}"
        return None

    def recode_religiosity(val):
        if pd.isna(val) or val in [77, 88, 99] or not (0 <= val <= 10):  return None
        if val <= 2:  return "Religiosity: Low (0-2)"
        if val <= 7:  return "Religiosity: Medium (3-7)"
        return "Religiosity: High (8-10)"

    def recode_internet_time(val):
        if pd.isna(val) or val in [6666, 7777, 8888, 9999] or val < 0 or val >= 6666:
            return None
        if val == 0:    return "Internet time: None"
        if val <= 30:   return "Internet time: 1-30 min"
        if val <= 60:   return "Internet time: 31-60 min"
        if val <= 180:  return "Internet time: 61-180 min"
        if val <= 300:  return "Internet time: 181-300 min"
        if val <= 420:  return "Internet time: 301-420 min"
        return "Internet time: 421+ min"

    def recode_domicile(val):
        return {1: "Big city", 2: "Suburbs/outskirts", 3: "Town/small city",
                4: "Country village", 5: "Farm/countryside"}.get(int(val)) if pd.notna(val) else None

    def recode_polintr(val):
        return {1: "Politics: Very interested", 2: "Politics: Quite interested",
                3: "Politics: Hardly interested",
                4: "Politics: Not at all interested"}.get(val)

    pol_vars = ["contplt", "donprty", "badge", "sgnptit", "pbldmna", "bctprd", "pstplonl"]

    def recode_pol_involvement(row):
        score, valid = 0, 0
        for v in pol_vars:
            val = row.get(v)
            if pd.notna(val) and val in [1, 2]:
                valid += 1
                if val == 1:
                    score += 1
        if valid < 4:  return None
        if score == 0:  return "Political involvement: None"
        if score == 1:  return "Political involvement: Low (1)"
        if score <= 3:  return "Political involvement: Moderate (2-3)"
        if score <= 5:  return "Political involvement: High (4-5)"
        return "Political involvement: Very high (6-7)"

    if "agea"    in df.columns: df["generation"]          = df["agea"].apply(recode_generation)
    if "edulvlb" in df.columns: df["education_bracket"]   = df["edulvlb"].apply(recode_education)
    if "nobingnd"in df.columns: df["gender_clean"]           = df["nobingnd"].apply(recode_gender)
    if "feethngr"in df.columns: df["ethnicmajority_clean"]   = df["feethngr"].apply(recode_ethnic)
    if "hinctnta"in df.columns: df["income_cat"]             = df["hinctnta"].apply(recode_income)
    if "rlgdgr"  in df.columns: df["religiosity_cat"]        = df["rlgdgr"].apply(recode_religiosity)
    if "netustm" in df.columns: df["internet_time_cat"]      = df["netustm"].apply(recode_internet_time)
    if "domicil" in df.columns: df["domicile"]               = df["domicil"].apply(recode_domicile)
    if "polintr" in df.columns: df["political_interest_cat"] = df["polintr"].apply(recode_polintr)
    if all(c in df.columns for c in pol_vars[:1]):
        df["political_involvement"] = df.apply(recode_pol_involvement, axis=1)
    if "hincfel" in df.columns: df["hincfel_clean"]          = df["hincfel"].apply(recode_hincfel)
    if "fnsdfml" in df.columns: df["fnsdfml_clean"]          = df["fnsdfml"].apply(recode_fnsdfml)
    if any(c in df.columns for c in ["rlgblg", "rlgdnm"]):
        df["religion_clean"] = df.apply(recode_religion, axis=1)
    for _mc in ["mnactic", "mainact"]:
        if _mc in df.columns:
            df["mnactic_clean"] = df[_mc].apply(recode_mnactic)
            break
    for _ic in ["iscoco", "isco08c", "isco08"]:
        if _ic in df.columns:
            df["isco08_major_group"] = df[_ic].apply(recode_isco)
            break
    if all(c in df.columns for c in ["brncntr", "facntr", "mocntr"]):
        df["immigration_background"] = df.apply(recode_immigration, axis=1)
    return df


def plot_heatmap_all_models(
    db_path,
    question_set_key: str,
    model_registry: dict,
    output_dir: Path,
    min_n: int = 30,
) -> None:
    """
    For each model: heatmap of (alignment − country mean) per subgroup × country.
    Subgroups are shown in a fixed order (_ALL_MODELS_CAT_GROUPS) with spacer
    rows between categories.
    Saved as all_models_{model_short}.png.
    """
    plt.style.use("default")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_full, model_short in model_registry.items():
        align_col = f"{model_short}_{question_set_key}_alignment"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(demographics)")
        demo_cols = {r[1] for r in cursor.fetchall()}
        if align_col not in demo_cols:
            conn.close()
            warnings.warn(f"Column '{align_col}' missing; skipping heatmap for {model_short}.")
            continue

        cursor.execute("PRAGMA table_info(ess11)")
        ess11_cols = {r[1] for r in cursor.fetchall()}

        base_ess = (
            "e.agea, e.edulvlb, e.nobingnd, e.feethngr, e.hinctnta, "
            "e.rlgdgr, e.netustm, e.domicil, e.polintr, "
            "e.contplt, e.donprty, e.badge, e.sgnptit, e.pbldmna, e.bctprd, e.pstplonl, "
            "e.brncntr, e.facntr, e.mocntr"
        )
        for _opt in ["hincfel", "fnsdfml", "rlgblg", "rlgdnm",
                     "mnactic", "mainact", "iscoco", "isco08c", "isco08",
                     "cntbrthd", "fbrncntc", "mbrncntc"]:
            if _opt in ess11_cols:
                base_ess += f", e.{_opt}"

        query = f"""
            SELECT d."{align_col}" AS alignment, e.cntry AS country, e.new_id,
                   {base_ess}
            FROM demographics d
            JOIN ess11 e ON d.new_id = e.new_id
            WHERE d."{align_col}" IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        df = _recode_all_models_heatmap(df)
        df = df.dropna(subset=["alignment", "country"])

        country_means = df.groupby("country")["alignment"].mean()
        country_order = country_means.sort_values(ascending=False).index.tolist()

        # Build pivot tables: subgroup × country
        all_rows, all_counts = [], []
        for cat_col, _ in _ALL_MODELS_CAT_GROUPS:
            if cat_col not in df.columns:
                continue
            grouped = (
                df.groupby(["country", cat_col])["alignment"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "aln", "count": "n"})
                .dropna(subset=[cat_col])
            )
            grouped["country_mean"] = grouped["country"].map(country_means)
            grouped["diff"] = grouped["aln"] - grouped["country_mean"]
            for _, r in grouped.iterrows():
                sg = str(r[cat_col])
                all_rows.append({"subgroup": sg, "country": r["country"], "diff": r["diff"]})
                all_counts.append({"subgroup": sg, "country": r["country"], "n": r["n"]})

        if not all_rows:
            continue

        pivot = pd.DataFrame(all_rows).pivot_table(
            index="subgroup", columns="country", values="diff", aggfunc="mean"
        )
        count_pivot = pd.DataFrame(all_counts).pivot_table(
            index="subgroup", columns="country", values="n", aggfunc="sum"
        )
        if pivot.empty:
            continue

        pivot       = pivot.reindex(columns=country_order, fill_value=np.nan)
        count_pivot = count_pivot.reindex(columns=country_order, fill_value=np.nan)

        # Build ordered rows with spacer rows between category groups
        ordered_rows = []   # list of (label, is_spacer)
        for cat_col, subgroups in _ALL_MODELS_CAT_GROUPS:
            present = [sg for sg in subgroups if sg in pivot.index]
            if not present:
                continue
            cat_label = _ALL_MODELS_CAT_LABELS.get(cat_col, cat_col.replace("_", " ").title())
            if ordered_rows:
                ordered_rows.append((cat_label, True))   # spacer with category header
            for sg in present:
                ordered_rows.append((sg, False))

        plot_data, count_data, plot_labels, spacer_rows = [], [], [], []
        for row_idx, (label, is_spacer) in enumerate(ordered_rows):
            if is_spacer:
                plot_data.append([np.nan] * len(country_order))
                count_data.append([np.nan] * len(country_order))
                plot_labels.append("")
                spacer_rows.append((row_idx, label))
            else:
                plot_data.append(pivot.loc[label, country_order].values.tolist())
                if label in count_pivot.index:
                    count_data.append(count_pivot.loc[label, country_order].values.tolist())
                else:
                    count_data.append([np.nan] * len(country_order))
                plot_labels.append(label)

        plot_df   = pd.DataFrame(plot_data,  columns=country_order, index=plot_labels)
        count_df2 = pd.DataFrame(count_data, columns=country_order, index=plot_labels)

        # Symmetric colour scale at 95th percentile
        all_vals = np.array(pivot.values.flatten(), dtype=float)
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals):
            pos = all_vals[all_vals > 0]
            neg = all_vals[all_vals < 0]
            vmax_abs = max(
                abs(np.percentile(pos, 95)) if len(pos) else 0.0,
                abs(np.percentile(neg,  5)) if len(neg) else 0.0,
            )
        else:
            vmax_abs = 0.05
        vmin, vmax = -vmax_abs, vmax_abs

        n_pr = len(plot_df)
        n_pc = len(country_order)
        fig, ax = plt.subplots(figsize=(max(16, n_pc * 0.55), max(10, n_pr * 0.32)))

        sns.heatmap(
            plot_df, ax=ax,
            cmap="RdBu", center=0, vmin=vmin, vmax=vmax,
            linewidths=0.5, linecolor="white",
            cbar_kws={
                "label":  f"Diff from country mean  (colour scale ±{vmax_abs:.3f})",
                "shrink": 0.4, "pad": 0.02,
            },
            mask=plot_df.isna(), annot=False,
        )

        # Hatch cells with N < 30 (black lines, slightly thicker)
        for row_idx in range(n_pr):
            for col_idx in range(n_pc):
                n_val = count_df2.iloc[row_idx, col_idx]
                if pd.notna(n_val) and n_val < 30:
                    ax.add_patch(plt.Rectangle(
                        (col_idx, row_idx), 1, 1,
                        fill=False, edgecolor="black", linewidth=1.8,
                        hatch="////", zorder=3, alpha=0.55,
                    ))

        # Draw category header labels in spacer rows
        for sp_idx, sp_label in spacer_rows:
            ax.text(n_pc / 2, sp_idx + 0.5, sp_label,
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color="#333333", zorder=5)

        # ax.set_title(
        #     f"{model_short}  [{question_set_key}] — Deviation from Country Mean Alignment Score\n"
        #     "Red = below country average  |  Blue = above country average  |  Hatch: N < 30",
        #     fontsize=16, fontweight="bold", pad=20,
        # )
        ax.set_xlabel("Country (ordered by average alignment)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Socio-Demographic Subgroup", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout(pad=2.0)

        out_path = output_dir / f"all_models_deviation_from_country_mean_{model_short}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Cross-model mean deviation heatmap  (subgroups × countries, mean across models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap_cross_model_mean(
    db_path,
    question_set_key: str,
    model_registry: dict,
    output_dir: Path,
    min_n: int = 30,
) -> None:
    """
    Single heatmap (subgroups × countries) showing the mean across all models
    of (group_alignment − country_mean_alignment).  Same layout as
    plot_heatmap_all_models but averaged over models rather than per-model.
    Cells with N < min_n are hatched in black.
    Saved as all_models_mean_deviation.png.
    """
    plt.style.use("default")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate per-model pivot tables
    pivot_stack: list[pd.DataFrame] = []
    count_pivot_ref: pd.DataFrame | None = None   # N counts (same for all models)
    country_order_ref: list[str] | None = None
    ordered_rows_ref: list[tuple] | None = None

    for model_full, model_short in model_registry.items():
        align_col = f"{model_short}_{question_set_key}_alignment"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(demographics)")
        demo_cols = {r[1] for r in cursor.fetchall()}
        if align_col not in demo_cols:
            conn.close()
            continue

        cursor.execute("PRAGMA table_info(ess11)")
        ess11_cols = {r[1] for r in cursor.fetchall()}
        base_ess = (
            "e.agea, e.edulvlb, e.nobingnd, e.feethngr, e.hinctnta, "
            "e.rlgdgr, e.netustm, e.domicil, e.polintr, "
            "e.contplt, e.donprty, e.badge, e.sgnptit, e.pbldmna, e.bctprd, e.pstplonl, "
            "e.brncntr, e.facntr, e.mocntr"
        )
        for _opt in ["hincfel", "fnsdfml", "rlgblg", "rlgdnm",
                     "mnactic", "mainact", "iscoco", "isco08c", "isco08",
                     "cntbrthd", "fbrncntc", "mbrncntc"]:
            if _opt in ess11_cols:
                base_ess += f", e.{_opt}"
        query = f"""
            SELECT d."{align_col}" AS alignment, e.cntry AS country, e.new_id,
                   {base_ess}
            FROM demographics d
            JOIN ess11 e ON d.new_id = e.new_id
            WHERE d."{align_col}" IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        df = _recode_all_models_heatmap(df)
        df = df.dropna(subset=["alignment", "country"])

        country_means = df.groupby("country")["alignment"].mean()
        if country_order_ref is None:
            country_order_ref = country_means.sort_values(ascending=False).index.tolist()

        all_rows, all_counts = [], []
        for cat_col, _ in _ALL_MODELS_CAT_GROUPS:
            if cat_col not in df.columns:
                continue
            grouped = (
                df.groupby(["country", cat_col])["alignment"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "aln", "count": "n"})
                .dropna(subset=[cat_col])
            )
            grouped["country_mean"] = grouped["country"].map(country_means)
            grouped["diff"] = grouped["aln"] - grouped["country_mean"]
            for _, r in grouped.iterrows():
                sg = str(r[cat_col])
                all_rows.append({"subgroup": sg, "country": r["country"], "diff": r["diff"]})
                all_counts.append({"subgroup": sg, "country": r["country"], "n": r["n"]})

        if not all_rows:
            continue

        pivot = pd.DataFrame(all_rows).pivot_table(
            index="subgroup", columns="country", values="diff", aggfunc="mean"
        ).reindex(columns=country_order_ref, fill_value=np.nan)
        pivot_stack.append(pivot)

        if count_pivot_ref is None:
            count_pivot_ref = (
                pd.DataFrame(all_counts)
                .pivot_table(index="subgroup", columns="country", values="n", aggfunc="sum")
                .reindex(columns=country_order_ref, fill_value=np.nan)
            )

        # Build ordered rows once
        if ordered_rows_ref is None:
            all_subgroups = pivot.index.tolist()
            ordered_rows_ref = []
            for cat_col, subgroups in _ALL_MODELS_CAT_GROUPS:
                present = [sg for sg in subgroups if sg in all_subgroups]
                if not present:
                    continue
                cat_label = _ALL_MODELS_CAT_LABELS.get(cat_col, cat_col.replace("_", " ").title())
                if ordered_rows_ref:
                    ordered_rows_ref.append((cat_label, True))
                for sg in present:
                    ordered_rows_ref.append((sg, False))

    if not pivot_stack or country_order_ref is None or ordered_rows_ref is None:
        warnings.warn("No data found; skipping cross-model mean deviation heatmap.")
        return

    # Average across models
    mean_pivot = pd.concat(pivot_stack).groupby(level=0).mean()
    mean_pivot = mean_pivot.reindex(columns=country_order_ref, fill_value=np.nan)

    # Assemble ordered rows
    plot_data, count_data, plot_labels, spacer_rows = [], [], [], []
    for row_idx, (label, is_spacer) in enumerate(ordered_rows_ref):
        if is_spacer:
            plot_data.append([np.nan] * len(country_order_ref))
            count_data.append([np.nan] * len(country_order_ref))
            plot_labels.append("")
            spacer_rows.append((row_idx, label))
        else:
            if label in mean_pivot.index:
                plot_data.append(mean_pivot.loc[label, country_order_ref].values.tolist())
            else:
                plot_data.append([np.nan] * len(country_order_ref))
            if count_pivot_ref is not None and label in count_pivot_ref.index:
                count_data.append(count_pivot_ref.loc[label, country_order_ref].values.tolist())
            else:
                count_data.append([np.nan] * len(country_order_ref))
            plot_labels.append(label)

    plot_df   = pd.DataFrame(plot_data,  columns=country_order_ref, index=plot_labels)
    count_df2 = pd.DataFrame(count_data, columns=country_order_ref, index=plot_labels)

    all_vals = np.array(mean_pivot.values.flatten(), dtype=float)
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals):
        pos = all_vals[all_vals > 0]
        neg = all_vals[all_vals < 0]
        vmax_abs = max(
            abs(np.percentile(pos, 95)) if len(pos) else 0.0,
            abs(np.percentile(neg,  5)) if len(neg) else 0.0,
        )
    else:
        vmax_abs = 0.05
    vmin, vmax = -vmax_abs, vmax_abs

    n_pr = len(plot_df)
    n_pc = len(country_order_ref)
    n_models_used = len(pivot_stack)

    fig, ax = plt.subplots(figsize=(max(16, n_pc * 0.55), max(10, n_pr * 0.32)))

    sns.heatmap(
        plot_df, ax=ax,
        cmap="RdBu", center=0, vmin=vmin, vmax=vmax,
        linewidths=0.5, linecolor="white",
        cbar_kws={
            "label":  f"Mean diff from country mean across {n_models_used} models  (±{vmax_abs:.3f})",
            "shrink": 0.4, "pad": 0.02,
        },
        mask=plot_df.isna(), annot=False,
    )

    # Hatch cells with N < min_n (black lines, slightly thicker)
    for row_idx in range(n_pr):
        for col_idx in range(n_pc):
            n_val = count_df2.iloc[row_idx, col_idx]
            if pd.notna(n_val) and n_val < min_n:
                ax.add_patch(plt.Rectangle(
                    (col_idx, row_idx), 1, 1,
                    fill=False, edgecolor="black", linewidth=1.8,
                    hatch="////", zorder=3, alpha=0.55,
                ))

    # Draw category header labels in spacer rows
    for sp_idx, sp_label in spacer_rows:
        ax.text(n_pc / 2, sp_idx + 0.5, sp_label,
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="#333333", zorder=5)

    # ax.set_title(
    #     f"[{question_set_key}]  Mean across {n_models_used} models — Deviation from Country Mean Alignment Score\n"
    #     "Red = below country average  |  Blue = above country average  |  Hatch: N < 30",
    #     fontsize=16, fontweight="bold", pad=20,
    # )
    ax.set_xlabel("Country (ordered by average alignment)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Socio-Demographic Subgroup", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout(pad=2.0)

    out_path = output_dir / "all_models_cross_model_mean_deviation_from_country_mean.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4c. Cross-model overall-mean deviation heatmap (subgroups × countries)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap_cross_model_overall(
    db_path,
    question_set_key: str,
    model_registry: dict,
    output_dir: Path,
    min_n: int = 30,
) -> None:
    """
    Single heatmap (subgroups × countries) showing the mean across all models
    of (group_alignment − model_overall_mean).  Identical layout to
    plot_heatmap_cross_model_mean but the baseline is the model-wide overall
    mean rather than the per-country mean.
    Countries on the x-axis are ordered by their mean deviation value.
    Cells with N < min_n are hatched in black.
    Saved as all_models_overall_mean_deviation.png.
    """
    plt.style.use("default")
    output_dir.mkdir(parents=True, exist_ok=True)

    pivot_stack: list[pd.DataFrame] = []
    count_pivot_ref: pd.DataFrame | None = None
    ordered_rows_ref: list[tuple] | None = None
    all_countries: set[str] = set()

    for model_full, model_short in model_registry.items():
        align_col = f"{model_short}_{question_set_key}_alignment"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(demographics)")
        demo_cols = {r[1] for r in cursor.fetchall()}
        if align_col not in demo_cols:
            conn.close()
            continue

        cursor.execute("PRAGMA table_info(ess11)")
        ess11_cols = {r[1] for r in cursor.fetchall()}
        base_ess = (
            "e.agea, e.edulvlb, e.nobingnd, e.feethngr, e.hinctnta, "
            "e.rlgdgr, e.netustm, e.domicil, e.polintr, "
            "e.contplt, e.donprty, e.badge, e.sgnptit, e.pbldmna, e.bctprd, e.pstplonl, "
            "e.brncntr, e.facntr, e.mocntr"
        )
        for _opt in ["hincfel", "fnsdfml", "rlgblg", "rlgdnm",
                     "mnactic", "mainact", "iscoco", "isco08c", "isco08",
                     "cntbrthd", "fbrncntc", "mbrncntc"]:
            if _opt in ess11_cols:
                base_ess += f", e.{_opt}"
        query = f"""
            SELECT d."{align_col}" AS alignment, e.cntry AS country, e.new_id,
                   {base_ess}
            FROM demographics d
            JOIN ess11 e ON d.new_id = e.new_id
            WHERE d."{align_col}" IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        df = _recode_all_models_heatmap(df)
        df = df.dropna(subset=["alignment", "country"])

        overall_mean = float(df["alignment"].mean())
        all_countries |= set(df["country"].unique())

        all_rows, all_counts = [], []
        for cat_col, _ in _ALL_MODELS_CAT_GROUPS:
            if cat_col not in df.columns:
                continue
            grouped = (
                df.groupby(["country", cat_col])["alignment"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "aln", "count": "n"})
                .dropna(subset=[cat_col])
            )
            grouped["diff"] = grouped["aln"] - overall_mean
            for _, r in grouped.iterrows():
                sg = str(r[cat_col])
                all_rows.append({"subgroup": sg, "country": r["country"], "diff": r["diff"]})
                all_counts.append({"subgroup": sg, "country": r["country"], "n": r["n"]})

        if not all_rows:
            continue

        pivot = pd.DataFrame(all_rows).pivot_table(
            index="subgroup", columns="country", values="diff", aggfunc="mean"
        )
        pivot_stack.append(pivot)

        if count_pivot_ref is None:
            count_pivot_ref = (
                pd.DataFrame(all_counts)
                .pivot_table(index="subgroup", columns="country", values="n", aggfunc="sum")
            )

        if ordered_rows_ref is None:
            all_subgroups = pivot.index.tolist()
            ordered_rows_ref = []
            for cat_col, subgroups in _ALL_MODELS_CAT_GROUPS:
                present = [sg for sg in subgroups if sg in all_subgroups]
                if not present:
                    continue
                cat_label = _ALL_MODELS_CAT_LABELS.get(cat_col, cat_col.replace("_", " ").title())
                if ordered_rows_ref:
                    ordered_rows_ref.append((cat_label, True))
                for sg in present:
                    ordered_rows_ref.append((sg, False))

    if not pivot_stack or ordered_rows_ref is None:
        warnings.warn("No data found; skipping cross-model overall deviation heatmap.")
        return

    # Average across models
    mean_pivot = pd.concat(pivot_stack).groupby(level=0).mean()

    # Order countries by their mean deviation across all subgroups (highest first)
    country_order = (
        mean_pivot.mean(axis=0)
        .sort_values(ascending=False)
        .index.tolist()
    )
    mean_pivot = mean_pivot.reindex(columns=country_order, fill_value=np.nan)
    if count_pivot_ref is not None:
        count_pivot_ref = count_pivot_ref.reindex(columns=country_order, fill_value=np.nan)

    # Assemble ordered rows
    plot_data, count_data, plot_labels, spacer_rows = [], [], [], []
    for row_idx, (label, is_spacer) in enumerate(ordered_rows_ref):
        if is_spacer:
            plot_data.append([np.nan] * len(country_order))
            count_data.append([np.nan] * len(country_order))
            plot_labels.append("")
            spacer_rows.append((row_idx, label))
        else:
            if label in mean_pivot.index:
                plot_data.append(mean_pivot.loc[label, country_order].values.tolist())
            else:
                plot_data.append([np.nan] * len(country_order))
            if count_pivot_ref is not None and label in count_pivot_ref.index:
                count_data.append(count_pivot_ref.loc[label, country_order].values.tolist())
            else:
                count_data.append([np.nan] * len(country_order))
            plot_labels.append(label)

    plot_df   = pd.DataFrame(plot_data,  columns=country_order, index=plot_labels)
    count_df2 = pd.DataFrame(count_data, columns=country_order, index=plot_labels)

    all_vals = np.array(mean_pivot.values.flatten(), dtype=float)
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals):
        pos = all_vals[all_vals > 0]
        neg = all_vals[all_vals < 0]
        vmax_abs = max(
            abs(np.percentile(pos, 95)) if len(pos) else 0.0,
            abs(np.percentile(neg,  5)) if len(neg) else 0.0,
        )
    else:
        vmax_abs = 0.05

    n_pr = len(plot_df)
    n_pc = len(country_order)
    n_models_used = len(pivot_stack)

    fig, ax = plt.subplots(figsize=(max(16, n_pc * 0.55), max(10, n_pr * 0.32)))

    sns.heatmap(
        plot_df, ax=ax,
        cmap="RdBu", center=0, vmin=-vmax_abs, vmax=vmax_abs,
        linewidths=0.5, linecolor="white",
        cbar_kws={
            "label":  f"Mean diff from overall mean across {n_models_used} models  (\u00b1{vmax_abs:.3f})",
            "shrink": 0.4, "pad": 0.02,
        },
        mask=plot_df.isna(), annot=False,
    )

    for row_idx in range(n_pr):
        for col_idx in range(n_pc):
            n_val = count_df2.iloc[row_idx, col_idx]
            if pd.notna(n_val) and n_val < min_n:
                ax.add_patch(plt.Rectangle(
                    (col_idx, row_idx), 1, 1,
                    fill=False, edgecolor="black", linewidth=1.8,
                    hatch="////", zorder=3, alpha=0.55,
                ))

    # Draw category header labels in spacer rows
    for sp_idx, sp_label in spacer_rows:
        ax.text(n_pc / 2, sp_idx + 0.5, sp_label,
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="#333333", zorder=5)

    # ax.set_title(
    #     f"[{question_set_key}]  Mean across {n_models_used} models — Deviation from Overall Mean Alignment Score\n"
    #     "Red = below overall average  |  Blue = above overall average  |  Hatch: N < 30",
    #     fontsize=16, fontweight="bold", pad=20,
    # )
    ax.set_xlabel("Country (ordered by mean deviation)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Socio-Demographic Subgroup", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout(pad=2.0)

    out_path = output_dir / "all_models_cross_model_mean_deviation_from_overall_mean.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 4d. Combined side-by-side: country-mean deviation | overall-mean deviation
# ─────────────────────────────────────────────────────────────────────────────

def _build_cross_model_pivot(
    db_path,
    question_set_key: str,
    model_registry: dict,
    baseline: str = "country",   # "country" or "overall"
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple], int]:
    """
    Shared data-builder for the three cross-model heatmap functions.

    Returns
    -------
    mean_pivot     : DataFrame  (subgroup index, country columns)
    count_pivot    : DataFrame  (same shape, sum of N counts)
    ordered_rows   : list of (label, is_spacer)
    n_models_used  : int
    """
    pivot_stack: list[pd.DataFrame] = []
    count_pivot_ref: pd.DataFrame | None = None
    ordered_rows_ref: list[tuple] | None = None
    country_order_ref: list[str] | None = None
    all_countries: set[str] = set()

    for model_full, model_short in model_registry.items():
        align_col = f"{model_short}_{question_set_key}_alignment"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(demographics)")
        demo_cols = {r[1] for r in cursor.fetchall()}
        if align_col not in demo_cols:
            conn.close()
            continue

        cursor.execute("PRAGMA table_info(ess11)")
        ess11_cols = {r[1] for r in cursor.fetchall()}
        base_ess = (
            "e.agea, e.edulvlb, e.nobingnd, e.feethngr, e.hinctnta, "
            "e.rlgdgr, e.netustm, e.domicil, e.polintr, "
            "e.contplt, e.donprty, e.badge, e.sgnptit, e.pbldmna, e.bctprd, e.pstplonl, "
            "e.brncntr, e.facntr, e.mocntr"
        )
        for _opt in ["hincfel", "fnsdfml", "rlgblg", "rlgdnm",
                     "mnactic", "mainact", "iscoco", "isco08c", "isco08",
                     "cntbrthd", "fbrncntc", "mbrncntc"]:
            if _opt in ess11_cols:
                base_ess += f", e.{_opt}"
        query = f"""
            SELECT d."{align_col}" AS alignment, e.cntry AS country, e.new_id,
                   {base_ess}
            FROM demographics d
            JOIN ess11 e ON d.new_id = e.new_id
            WHERE d."{align_col}" IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        df = _recode_all_models_heatmap(df)
        df = df.dropna(subset=["alignment", "country"])

        country_means = df.groupby("country")["alignment"].mean()
        overall_mean  = float(df["alignment"].mean())
        all_countries |= set(df["country"].unique())

        if baseline == "country" and country_order_ref is None:
            country_order_ref = country_means.sort_values(ascending=False).index.tolist()

        all_rows, all_counts = [], []
        for cat_col, _ in _ALL_MODELS_CAT_GROUPS:
            if cat_col not in df.columns:
                continue
            grouped = (
                df.groupby(["country", cat_col])["alignment"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "aln", "count": "n"})
                .dropna(subset=[cat_col])
            )
            if baseline == "country":
                grouped["country_mean"] = grouped["country"].map(country_means)
                grouped["diff"] = grouped["aln"] - grouped["country_mean"]
            else:
                grouped["diff"] = grouped["aln"] - overall_mean
            for _, r in grouped.iterrows():
                sg = str(r[cat_col])
                all_rows.append({"subgroup": sg, "country": r["country"], "diff": r["diff"]})
                all_counts.append({"subgroup": sg, "country": r["country"], "n": r["n"]})

        if not all_rows:
            continue

        pivot = pd.DataFrame(all_rows).pivot_table(
            index="subgroup", columns="country", values="diff", aggfunc="mean"
        )
        pivot_stack.append(pivot)

        if count_pivot_ref is None:
            count_pivot_ref = (
                pd.DataFrame(all_counts)
                .pivot_table(index="subgroup", columns="country", values="n", aggfunc="sum")
            )

        if ordered_rows_ref is None:
            all_subgroups = pivot.index.tolist()
            ordered_rows_ref = []
            for cat_col, subgroups in _ALL_MODELS_CAT_GROUPS:
                present = [sg for sg in subgroups if sg in all_subgroups]
                if not present:
                    continue
                cat_label = _ALL_MODELS_CAT_LABELS.get(cat_col, cat_col.replace("_", " ").title())
                if ordered_rows_ref:
                    ordered_rows_ref.append((cat_label, True))
                for sg in present:
                    ordered_rows_ref.append((sg, False))

    if not pivot_stack or ordered_rows_ref is None:
        return pd.DataFrame(), pd.DataFrame(), [], 0

    mean_pivot = pd.concat(pivot_stack).groupby(level=0).mean()

    if baseline == "country":
        country_order = country_order_ref or (
            mean_pivot.mean(axis=0).sort_values(ascending=False).index.tolist()
        )
    else:
        country_order = (
            mean_pivot.mean(axis=0).sort_values(ascending=False).index.tolist()
        )

    mean_pivot = mean_pivot.reindex(columns=country_order, fill_value=np.nan)
    if count_pivot_ref is not None:
        count_pivot_ref = count_pivot_ref.reindex(columns=country_order, fill_value=np.nan)
    else:
        count_pivot_ref = pd.DataFrame(
            np.nan, index=mean_pivot.index, columns=mean_pivot.columns
        )

    return mean_pivot, count_pivot_ref, ordered_rows_ref, len(pivot_stack)


def plot_heatmap_cross_model_combined(
    db_path,
    question_set_key: str,
    model_registry: dict,
    output_dir: Path,
    min_n: int = 30,
) -> None:
    """
    Side-by-side heatmap:
      Left  — deviation from country mean (same as all_models_mean_deviation)
      Right — deviation from overall mean  (same as all_models_overall_mean_deviation)

    Shares a single colour scale, y-axis tick labels only on the left panel.
    Saved as all_models_combined_deviation.png.
    """
    plt.style.use("default")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build both pivot tables
    cntry_pivot, cntry_counts, ordered_rows, n_models = _build_cross_model_pivot(
        db_path, question_set_key, model_registry, baseline="country"
    )
    overall_pivot, overall_counts, _, _ = _build_cross_model_pivot(
        db_path, question_set_key, model_registry, baseline="overall"
    )

    if cntry_pivot.empty or overall_pivot.empty:
        warnings.warn("No data; skipping combined cross-model heatmap.")
        return

    # Force both panels to share the left panel's country order
    country_order  = cntry_pivot.columns.tolist()
    overall_pivot  = overall_pivot.reindex(columns=country_order, fill_value=np.nan)
    overall_counts = overall_counts.reindex(columns=country_order, fill_value=np.nan)

    # Assemble ordered row arrays for both panels
    def _assemble(pivot, counts, ordered_rows):
        countries = pivot.columns.tolist()
        plot_data, count_data, plot_labels, spacer_rows = [], [], [], []
        for row_idx, (label, is_spacer) in enumerate(ordered_rows):
            if is_spacer:
                plot_data.append([np.nan] * len(countries))
                count_data.append([np.nan] * len(countries))
                plot_labels.append("")
                spacer_rows.append((row_idx, label))
            else:
                plot_data.append(
                    pivot.loc[label, countries].values.tolist()
                    if label in pivot.index else [np.nan] * len(countries)
                )
                count_data.append(
                    counts.loc[label, countries].values.tolist()
                    if label in counts.index else [np.nan] * len(countries)
                )
                plot_labels.append(label)
        return (
            pd.DataFrame(plot_data,  columns=countries, index=plot_labels),
            pd.DataFrame(count_data, columns=countries, index=plot_labels),
            spacer_rows,
        )

    cntry_df,   cntry_cnt_df,   spacer_rows_c = _assemble(cntry_pivot,   cntry_counts,   ordered_rows)
    overall_df, overall_cnt_df, spacer_rows_o = _assemble(overall_pivot, overall_counts, ordered_rows)

    # Shared symmetric colour scale (95th-percentile clipping)
    all_vals = np.concatenate([
        cntry_pivot.values.flatten(),
        overall_pivot.values.flatten(),
    ])
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals):
        pos = all_vals[all_vals > 0]
        neg = all_vals[all_vals < 0]
        vmax_abs = max(
            abs(np.percentile(pos, 95)) if len(pos) else 0.0,
            abs(np.percentile(neg,  5)) if len(neg) else 0.0,
        )
    else:
        vmax_abs = 0.05
    if vmax_abs == 0:
        vmax_abs = 0.05

    n_pr   = len(cntry_df)
    n_pc_c = len(cntry_pivot.columns)
    n_pc_o = len(overall_pivot.columns)

    # ── Figure layout with square cells and equal-size panels ────────────────
    cell_size  = 0.45   # inches per cell (square)
    pad_left   = 5.5    # space for y-tick labels
    pad_bottom = 3.2    # space for rotated x-tick labels
    pad_top    = 0.4
    pad_mid    = 0.9    # gap between panels
    cbar_gap   = 0.25   # gap: right panel → colorbar
    cbar_w_in  = 0.45   # colorbar width
    cbar_r_in  = 1.4    # right margin after colorbar

    panel_h_in = n_pr   * cell_size
    panel_w_l  = n_pc_c * cell_size
    panel_w_r  = n_pc_o * cell_size

    fig_w = pad_left + panel_w_l + pad_mid + panel_w_r + cbar_gap + cbar_w_in + cbar_r_in
    fig_h = pad_top  + panel_h_in + pad_bottom

    fig = plt.figure(figsize=(fig_w, fig_h))

    def _fx(x): return x / fig_w
    def _fy(y): return y / fig_h

    ax_l = fig.add_axes([
        _fx(pad_left),           _fy(pad_bottom),
        _fx(panel_w_l),          _fy(panel_h_in),
    ])
    ax_r = fig.add_axes([
        _fx(pad_left + panel_w_l + pad_mid),  _fy(pad_bottom),
        _fx(panel_w_r),                        _fy(panel_h_in),
    ])
    cbar_ax = fig.add_axes([
        _fx(pad_left + panel_w_l + pad_mid + panel_w_r + cbar_gap),
        _fy(pad_bottom + panel_h_in * 0.2),
        _fx(cbar_w_in),
        _fy(panel_h_in * 0.6),
    ])

    # ── Left panel (country-mean deviation) ──────────────────────────────────
    sns.heatmap(
        cntry_df, ax=ax_l,
        cmap="RdBu", center=0, vmin=-vmax_abs, vmax=vmax_abs,
        linewidths=0.5, linecolor="white",
        cbar=False,
        mask=cntry_df.isna(), annot=False,
    )
    _spacer_rows_c = {sp_idx for sp_idx, _ in spacer_rows_c}
    for row_idx in range(n_pr):
        if row_idx in _spacer_rows_c:
            continue
        for col_idx in range(n_pc_c):
            cnt = cntry_cnt_df.iloc[row_idx, col_idx]
            val = cntry_df.iloc[row_idx, col_idx]
            if pd.isna(val) or (pd.notna(cnt) and cnt < min_n) or (pd.notna(cnt) and cnt == 0):
                ax_l.add_patch(plt.Rectangle(
                    (col_idx, row_idx), 1, 1,
                    fill=True, facecolor="#808080", edgecolor="none",
                    zorder=3, alpha=1.0,
                ))
    for sp_idx, sp_label in spacer_rows_c:
        ax_l.text(n_pc_c / 2, sp_idx + 0.5, sp_label,
                  ha="center", va="center",
                  fontsize=20, fontweight="bold", color="#333333", zorder=5)
    ax_l.set_xlabel("Country (ordered by avg. alignment)", fontsize=20, fontweight="bold")
    ax_l.set_ylabel("Socio-Demographic Subgroup", fontsize=22, fontweight="bold")
    ax_l.tick_params(axis="x", rotation=45, labelsize=17)
    ax_l.tick_params(axis="y", labelsize=17)
    ax_l.set_xticklabels(ax_l.get_xticklabels(), ha="right")

    # ── Right panel (overall-mean deviation) ─────────────────────────────────
    sns.heatmap(
        overall_df, ax=ax_r,
        cmap="RdBu", center=0, vmin=-vmax_abs, vmax=vmax_abs,
        linewidths=0.5, linecolor="white",
        cbar=False,
        mask=overall_df.isna(), annot=False,
    )
    _spacer_rows_o = {sp_idx for sp_idx, _ in spacer_rows_o}
    for row_idx in range(n_pr):
        if row_idx in _spacer_rows_o:
            continue
        for col_idx in range(n_pc_o):
            cnt = overall_cnt_df.iloc[row_idx, col_idx]
            val = overall_df.iloc[row_idx, col_idx]
            if pd.isna(val) or (pd.notna(cnt) and cnt < min_n) or (pd.notna(cnt) and cnt == 0):
                ax_r.add_patch(plt.Rectangle(
                    (col_idx, row_idx), 1, 1,
                    fill=True, facecolor="#808080", edgecolor="none",
                    zorder=3, alpha=1.0,
                ))
    for sp_idx, sp_label in spacer_rows_o:
        ax_r.text(n_pc_o / 2, sp_idx + 0.5, sp_label,
                  ha="center", va="center",
                  fontsize=20, fontweight="bold", color="#333333", zorder=5)
    ax_r.set_xlabel("Country (ordered by avg. alignment)", fontsize=20, fontweight="bold")
    ax_r.set_ylabel("")
    ax_r.tick_params(axis="y", left=False, labelleft=False)
    ax_r.tick_params(axis="x", rotation=45, labelsize=17)
    ax_r.set_xticklabels(ax_r.get_xticklabels(), ha="right")

    # ── Colorbar (manually placed, right of right panel) ─────────────────────
    import matplotlib.colors as _mcolors
    import matplotlib.cm as _cm
    norm = _mcolors.TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)
    sm   = _cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.ax.set_ylabel("Deviation from mean", fontsize=18, labelpad=12)
    cb.ax.tick_params(labelsize=16)

    out_path = output_dir / "all_models_combined_country_mean_and_overall_mean_deviation.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path} (+ .pdf)")


# ─────────────────────────────────────────────────────────────────────────────

def plot_model_correlation_heatmap(
    db_path,
    question_set_key: str,
    model_registry: dict,
    output_dir: Path,
    majority_votes_csvs: "list | None" = None,
) -> None:
    """
    Pairwise Pearson correlation between models, computed over each model's
    majority-vote answer per survey question (from majority_votes_csvs if
    provided; falls back to mean alignment per question from the per_question
    DB table if no majority-vote data is available).  Uses the intersection of
    questions answered by all models (= largest shared set).  Cells are
    annotated with r and significance stars.
    Saved as model_correlation_pairwise_question_level_majority_vote.png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    conn  = sqlite3.connect(db_path)
    avail = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"].tolist()

    # Per-model majority-vote answer per question
    model_values: dict[str, dict[str, float]] = {}

    if majority_votes_csvs:
        # Determine which question_vars belong to this question set by
        # inspecting the per_question table columns (if it exists).
        table = f"per_question_{question_set_key}"
        qs_vars: set[str] | None = None
        if table in avail:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info(\"{table}\")")
            cols = [r[1] for r in cursor.fetchall()]
            all_short = set(model_registry.values())
            qs_vars = set()
            for col in cols:
                for short in all_short:
                    if col.startswith(f"{short}_"):
                        qs_vars.add(col[len(short) + 1:])
                        break

        dfs_mv = []
        for p in (majority_votes_csvs if isinstance(majority_votes_csvs, list) else [majority_votes_csvs]):
            try:
                dfs_mv.append(pd.read_csv(p, usecols=["model", "question_var", "majority_vote_answer"]))
            except Exception:
                pass
        if dfs_mv:
            df_mv = pd.concat(dfs_mv, ignore_index=True)
            df_mv["majority_vote_answer"] = pd.to_numeric(df_mv["majority_vote_answer"], errors="coerce")
            if qs_vars:
                df_mv = df_mv[df_mv["question_var"].isin(qs_vars)]
            for model_full, model_short in model_registry.items():
                sub = df_mv[df_mv["model"] == model_full][["question_var", "majority_vote_answer"]].dropna()
                if not sub.empty:
                    model_values[model_short] = dict(zip(sub["question_var"], sub["majority_vote_answer"]))

    # Fallback: mean alignment per question from DB table
    if not model_values:
        table = f"per_question_{question_set_key}"
        if table not in avail:
            conn.close()
            warnings.warn(f"Table '{table}' not found; skipping correlation heatmap.")
            return

        df_pq = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        for _, model_short in model_registry.items():
            prefix = f"{model_short}_"
            q_cols = {c[len(prefix):]: c for c in df_pq.columns if c.startswith(prefix)}
            if not q_cols:
                continue
            means: dict[str, float] = {}
            for var, col in q_cols.items():
                vals = pd.to_numeric(df_pq[col], errors="coerce").dropna()
                if len(vals):
                    means[var] = float(vals.mean())
            if means:
                model_values[model_short] = means

    conn.close()

    if len(model_values) < 2:
        warnings.warn("Need at least 2 models for correlation heatmap; skipping.")
        return

    # Intersection: questions with data for every model
    df_in = pd.DataFrame(model_values).dropna()
    n_q   = len(df_in)
    if n_q < 3:
        warnings.warn(f"Only {n_q} shared questions; skipping correlation heatmap.")
        return

    print(f"  Correlation heatmap: {n_q} shared questions × {len(model_values)} models")

    models   = list(model_values.keys())
    n_models = len(models)

    # Pearson r + p-values
    try:
        from scipy.stats import pearsonr
        use_scipy = True
    except ImportError:
        use_scipy = False

    corr = np.zeros((n_models, n_models))
    pval = np.zeros((n_models, n_models))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                corr[i, j] = 1.0
            else:
                x = df_in[m1].values
                y = df_in[m2].values
                if use_scipy:
                    r, p = pearsonr(x, y)
                else:
                    r = float(np.corrcoef(x, y)[0, 1])
                    # approximate p-value via t-distribution
                    t = r * np.sqrt((n_q - 2) / max(1 - r ** 2, 1e-12))
                    from scipy.stats import t as t_dist  # try anyway
                    p = 2 * (1 - t_dist.cdf(abs(t), df=n_q - 2)) if use_scipy else float("nan")
                corr[i, j] = r
                pval[i, j] = p

    df_corr = pd.DataFrame(corr, index=models, columns=models)

    # Annotation strings
    annot = np.empty_like(corr, dtype=object)
    for i in range(n_models):
        for j in range(n_models):
            r = corr[i, j]
            p = pval[i, j]
            if i == j:
                annot[i, j] = "1.000"
            else:
                stars = (
                    "***" if p < 0.001 else
                    "**"  if p < 0.01  else
                    "*"   if p < 0.05  else ""
                )
                annot[i, j] = f"{r:.3f}{stars}"

    # Draw
    fig_sz = max(5, n_models * 1.8)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))

    sns.heatmap(
        df_corr, ax=ax,
        cmap="RdYlGn", vmin=-1, vmax=1, center=0,
        annot=annot, fmt="",
        annot_kws={"size": 12, "fontweight": "bold"},
        linewidths=2, linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.6},
        square=True,
    )

    qs_label = f"  [{question_set_key}]" if question_set_key else ""

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=13)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=13)

    plt.tight_layout()
    out_path = output_dir / "model_correlation_pairwise_question_level_majority_vote.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_model_correlation_heatmap_person(
    db_path,
    question_set_key: str,
    model_registry: dict,
    output_dir: Path,
) -> None:
    """
    Pairwise Pearson correlation between models, computed over person-level
    alignment scores.  Uses the intersection of respondents with a score from
    every model.  Cells are annotated with r and significance stars.
    Saved as model_correlation_heatmap_person.png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    conn  = sqlite3.connect(db_path)
    avail = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"].tolist()
    if "demographics" not in avail:
        conn.close()
        warnings.warn("Table 'demographics' not found; skipping person-level correlation heatmap.")
        return

    df_demo = pd.read_sql_query("SELECT * FROM demographics", conn)
    conn.close()

    # Collect per-model alignment column
    model_cols: dict[str, str] = {}
    for _, model_short in model_registry.items():
        col = f"{model_short}_{question_set_key}_alignment"
        if col in df_demo.columns:
            model_cols[model_short] = col

    if len(model_cols) < 2:
        warnings.warn("Need at least 2 models with alignment columns; skipping person-level correlation heatmap.")
        return

    # Build person × model matrix, drop anyone missing any model
    df_in = pd.DataFrame(
        {m: pd.to_numeric(df_demo[col], errors="coerce") for m, col in model_cols.items()}
    ).dropna()
    n_persons = len(df_in)
    if n_persons < 3:
        warnings.warn(f"Only {n_persons} shared respondents; skipping person-level correlation heatmap.")
        return

    print(f"  Person-level correlation heatmap: {n_persons} respondents × {len(model_cols)} models")

    models   = list(model_cols.keys())
    n_models = len(models)

    try:
        from scipy.stats import pearsonr
        use_scipy = True
    except ImportError:
        use_scipy = False

    corr = np.zeros((n_models, n_models))
    pval = np.zeros((n_models, n_models))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                corr[i, j] = 1.0
            else:
                x = df_in[m1].values
                y = df_in[m2].values
                if use_scipy:
                    r, p = pearsonr(x, y)
                else:
                    r = float(np.corrcoef(x, y)[0, 1])
                    t = r * np.sqrt((n_persons - 2) / max(1 - r ** 2, 1e-12))
                    p = float("nan")
                corr[i, j] = r
                pval[i, j] = p

    df_corr = pd.DataFrame(corr, index=models, columns=models)

    annot = np.empty_like(corr, dtype=object)
    for i in range(n_models):
        for j in range(n_models):
            r = corr[i, j]
            p = pval[i, j]
            if i == j:
                annot[i, j] = "1.000"
            else:
                stars = (
                    "***" if p < 0.001 else
                    "**"  if p < 0.01  else
                    "*"   if p < 0.05  else ""
                )
                annot[i, j] = f"{r:.3f}{stars}"

    fig_sz = max(5, n_models * 1.8)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))

    sns.heatmap(
        df_corr, ax=ax,
        cmap="RdYlGn", vmin=-1, vmax=1, center=0,
        annot=annot, fmt="",
        annot_kws={"size": 12, "fontweight": "bold"},
        linewidths=2, linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.6},
        square=True,
    )

    qs_label = f"  [{question_set_key}]" if question_set_key else ""

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=13)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=13)

    plt.tight_layout()
    out_path = output_dir / "model_correlation_pairwise_person_level.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_model_correlation_heatmap_combined(
    db_path,
    question_set_key: str,
    model_registry: dict,
    output_dir: Path,
    majority_votes_csvs: "list | None" = None,
) -> None:
    """
    Combined pairwise correlation heatmap:
      - Upper triangle: Pearson correlations over each model's majority-vote
        answer per survey question (from majority_votes_csvs if provided;
        falls back to mean alignment per question from the per_question DB
        table if no majority-vote data is available).
      - Lower triangle: Pearson correlations over per-respondent alignment
        scores (person-level).
      - Diagonal: 1.000
    A dashed diagonal line separates the two triangles.
    Saved as model_correlation_pairwise_question_and_person_level_combined.png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from scipy.stats import pearsonr
        use_scipy = True
    except ImportError:
        use_scipy = False

    conn  = sqlite3.connect(db_path)
    avail = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"].tolist()

    # ── Question-level data (majority-vote answers per question) ──────────────
    q_model_values: dict[str, dict[str, float]] = {}

    if majority_votes_csvs:
        # Determine which question_vars belong to this question set by
        # inspecting the per_question table columns (if it exists).
        table = f"per_question_{question_set_key}"
        qs_vars: set[str] | None = None
        if table in avail:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info(\"{table}\")")
            cols = [r[1] for r in cursor.fetchall()]
            # Columns are like "{model_short}_{question_var}"; extract unique vars.
            all_short = set(model_registry.values())
            qs_vars = set()
            for col in cols:
                for short in all_short:
                    if col.startswith(f"{short}_"):
                        qs_vars.add(col[len(short) + 1:])
                        break

        # Load and concatenate majority-votes CSVs.
        dfs_mv = []
        for p in (majority_votes_csvs if isinstance(majority_votes_csvs, list) else [majority_votes_csvs]):
            try:
                dfs_mv.append(pd.read_csv(p, usecols=["model", "question_var", "majority_vote_answer"]))
            except Exception:
                pass
        if dfs_mv:
            df_mv = pd.concat(dfs_mv, ignore_index=True)
            df_mv["majority_vote_answer"] = pd.to_numeric(df_mv["majority_vote_answer"], errors="coerce")
            if qs_vars:
                df_mv = df_mv[df_mv["question_var"].isin(qs_vars)]
            # Build reverse map: model_full → model_short
            rev_registry = {v: k for k, v in model_registry.items()}
            for model_full, model_short in model_registry.items():
                sub = df_mv[df_mv["model"] == model_full][["question_var", "majority_vote_answer"]].dropna()
                if not sub.empty:
                    q_model_values[model_short] = dict(zip(sub["question_var"], sub["majority_vote_answer"]))

    # Fallback: mean alignment per question from DB table
    if not q_model_values:
        table = f"per_question_{question_set_key}"
        if table in avail:
            df_pq = pd.read_sql_query(f"SELECT * FROM \"{table}\"", conn)
            for _, model_short in model_registry.items():
                prefix = f"{model_short}_"
                q_cols = {c[len(prefix):]: c for c in df_pq.columns if c.startswith(prefix)}
                if not q_cols:
                    continue
                means: dict[str, float] = {}
                for var, col in q_cols.items():
                    vals = pd.to_numeric(df_pq[col], errors="coerce").dropna()
                    if len(vals):
                        means[var] = float(vals.mean())
                if means:
                    q_model_values[model_short] = means

    # ── Person-level data ─────────────────────────────────────────────────────
    p_model_cols: dict[str, str] = {}
    df_demo: pd.DataFrame | None = None
    if "demographics" in avail:
        df_demo = pd.read_sql_query("SELECT * FROM demographics", conn)
        for _, model_short in model_registry.items():
            col = f"{model_short}_{question_set_key}_alignment"
            if col in df_demo.columns:
                p_model_cols[model_short] = col

    conn.close()

    # Common models present in both datasets
    common_models = sorted(set(q_model_values.keys()) & set(p_model_cols.keys()))
    if len(common_models) < 2:
        warnings.warn(
            "Need at least 2 models with both question- and person-level data; "
            "skipping combined heatmap."
        )
        return

    df_q_in = pd.DataFrame({m: q_model_values[m] for m in common_models}).dropna()
    n_q = len(df_q_in)
    if n_q < 3:
        warnings.warn(f"Only {n_q} shared questions; skipping combined heatmap.")
        return

    df_p_in = pd.DataFrame(
        {m: pd.to_numeric(df_demo[p_model_cols[m]], errors="coerce") for m in common_models}
    ).dropna()
    n_persons = len(df_p_in)
    if n_persons < 3:
        warnings.warn(f"Only {n_persons} shared respondents; skipping combined heatmap.")
        return

    print(
        f"  Combined correlation heatmap: {n_q} shared questions, "
        f"{n_persons:,} respondents × {len(common_models)} models"
    )

    n_models = len(common_models)

    def _compute_corr(df: pd.DataFrame):
        corr = np.zeros((n_models, n_models))
        pval = np.zeros((n_models, n_models))
        n    = len(df)
        for i, m1 in enumerate(common_models):
            for j, m2 in enumerate(common_models):
                if i == j:
                    corr[i, j] = 1.0
                else:
                    x = df[m1].values
                    y = df[m2].values
                    if use_scipy:
                        r, p = pearsonr(x, y)
                    else:
                        r = float(np.corrcoef(x, y)[0, 1])
                        p = float("nan")
                    corr[i, j] = r
                    pval[i, j] = p
        return corr, pval

    q_corr, q_pval = _compute_corr(df_q_in)
    p_corr, p_pval = _compute_corr(df_p_in)

    def _stars(p: float) -> str:
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    # ── Build combined matrices ───────────────────────────────────────────────
    combined_corr  = np.zeros((n_models, n_models))
    combined_annot = np.empty((n_models, n_models), dtype=object)

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                combined_corr[i, j]  = 1.0
                combined_annot[i, j] = "1.000"
            elif i > j:   # lower triangle → person-level
                r = p_corr[i, j]
                combined_corr[i, j]  = r
                combined_annot[i, j] = f"{r:.3f}"
            else:          # upper triangle → question-level
                r = q_corr[i, j]
                combined_corr[i, j]  = r
                combined_annot[i, j] = f"{r:.3f}"

    df_combined = pd.DataFrame(combined_corr, index=common_models, columns=common_models)
    diag_mask = np.eye(n_models, dtype=bool)

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig_sz = max(5, n_models * 2.4)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))

    sns.heatmap(
        df_combined, ax=ax,
        mask=diag_mask,
        cmap="RdBu", vmin=-1, vmax=1, center=0,
        annot=combined_annot, fmt="",
        annot_kws={"size": 24, "fontweight": "bold"},
        linewidths=2, linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.6},
        square=True,
    )

    # Increase colorbar label and tick sizes
    cbar = ax.collections[0].colorbar
    cbar.set_label("Pearson r", fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    # Diagonal separator line (top-left → bottom-right of heatmap area)
    ax.plot([0, n_models], [0, n_models], color="#8b0000", lw=2.5, ls="--", zorder=5)

    qs_label  = f"  [{question_set_key}]" if question_set_key else ""
    star_note = "  |  * p<.05  ** p<.01  *** p<.001" if use_scipy else ""
    # ax.set_title(
    #     f"Pairwise Model Correlation{qs_label}\n"
    #     f"Upper triangle: majority-vote answers ({n_q} questions)  ·  "
    #     f"Lower triangle: person-level alignment ({n_persons:,} respondents)"
    #     + star_note,
    #     fontsize=14, fontweight="bold", pad=15,
    # )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=22)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=22)

    plt.tight_layout()
    out_path = output_dir / "model_correlation_pairwise_question_and_person_level_combined.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path} (+ .pdf)")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Cross-model mean deviation plots (3 variants)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_model_mean_deviation(
    bootstrap_results_dir: Path,
    output_dir: Path,
    question_set_key: str = "",
    min_n_respondents: int = 100,
    conf_level: float = 0.95,
    variants: tuple[str, ...] = ("means", "ci", "models"),
) -> None:
    """
    Up to three figures showing the cross-model average of
    (group_alignment − model_overall_mean) for every SD group and country.
    Which figures are produced is controlled by *variants*:

    Variant "means"  – means only         → cross_model_deviation_means.png
    Variant "ci"     – means + CI         → cross_model_deviation_ci.png
    Variant "models" – means + CI + per-model dots → cross_model_deviation_models.png

    CI is propagated from per-model bootstrap SEs:
        se_combined = sqrt(Σ se_i²) / n_models
    where se_i = (ci_upper_i − ci_lower_i) / (2 · z_{α/2}).
    """
    from scipy.stats import norm as _norm

    output_dir.mkdir(parents=True, exist_ok=True)
    df_overall, var_dfs = _load_bootstrap_results(bootstrap_results_dir)

    if df_overall is None or df_overall.empty:
        warnings.warn("No overall alignment data; skipping cross-model deviation plots.")
        return

    all_models = sorted(df_overall["model"].unique().tolist())
    if not all_models:
        return

    z_alpha = float(_norm.ppf((1 + conf_level) / 2))

    # Overall mean per model
    overall: dict[str, float] = {
        row["model"]: float(row["mean_alignment"])
        for _, row in df_overall.iterrows()
    }

    # ── Build ordered row list (SD vars then countries) ───────────────────────
    sd_rows: list[tuple[str, object]] = []
    for var in _MASTER_VAR_ORDER:
        if var not in var_dfs:
            continue
        df_v = var_dfs[var]
        valid_groups: set = set()
        for m in all_models:
            sub = df_v[(df_v["model"] == m) & (df_v["n_respondents_mean"] >= min_n_respondents)]
            valid_groups |= set(sub["group"].unique())
        for g in _sort_groups(list(valid_groups), var):
            sd_rows.append((var, g))

    cntry_df = var_dfs.get("cntry")
    country_rows: list[str] = []
    if cntry_df is not None:
        cf = cntry_df[cntry_df["n_respondents_mean"] >= min_n_respondents]
        cmeans = cf.groupby("group")["mean_alignment"].mean()
        country_rows = cmeans.sort_values(ascending=True).index.tolist()

    if not sd_rows and not country_rows:
        warnings.warn("No data rows; skipping cross-model deviation plots.")
        return

    # ── Compute per-row deviation stats ───────────────────────────────────────
    # Prefer pre-computed bootstrap deviation CSVs (correct method).
    # Fall back to SE propagation only if they don't exist.
    dev_dfs = _load_deviation_results(bootstrap_results_dir)
    use_precomputed = bool(dev_dfs)

    all_rows_meta: list[dict] = []
    for var, group in sd_rows:
        df_v = var_dfs[var]

        if use_precomputed and var in dev_dfs:
            ddf = dev_dfs[var]
            row_d = ddf[ddf["group"] == group]
            if row_d.empty:
                continue
            r = row_d.iloc[0]
            mn   = float(r["mean_dev"])
            ci_lo = float(r["ci_lower_dev"])
            ci_hi = float(r["ci_upper_dev"])
            # Still collect per-model deviations for variant-C dots
            per_model: list[tuple[str, float]] = []
            for m in all_models:
                if m not in overall:
                    continue
                sub = df_v[
                    (df_v["model"] == m) & (df_v["group"] == group) &
                    (df_v["n_respondents_mean"] >= min_n_respondents)
                ]
                if sub.empty:
                    continue
                per_model.append((m, float(sub.iloc[0]["mean_alignment"]) - overall[m]))
            all_rows_meta.append(dict(
                kind="sd", var=var, group=group,
                mean_dev=mn, ci_lo=ci_lo, ci_hi=ci_hi,
                n_models=len(per_model), per_model=per_model,
            ))
        else:
            # Fallback: propagate SEs
            devs, ses = [], []
            per_model = []
            for m in all_models:
                if m not in overall:
                    continue
                sub = df_v[
                    (df_v["model"] == m) & (df_v["group"] == group) &
                    (df_v["n_respondents_mean"] >= min_n_respondents)
                ]
                if sub.empty:
                    continue
                r   = sub.iloc[0]
                dev = float(r["mean_alignment"]) - overall[m]
                se  = (float(r["ci_upper"]) - float(r["ci_lower"])) / (2 * z_alpha)
                devs.append(dev)
                ses.append(se)
                per_model.append((m, dev))
            if not devs:
                continue
            n    = len(devs)
            mn   = float(np.mean(devs))
            se_c = float(np.sqrt(np.sum(np.array(ses) ** 2))) / n
            all_rows_meta.append(dict(
                kind="sd", var=var, group=group,
                mean_dev=mn, ci_lo=mn - z_alpha * se_c, ci_hi=mn + z_alpha * se_c,
                n_models=n, per_model=per_model,
            ))

    for c in country_rows:
        if cntry_df is None:
            continue

        if use_precomputed and "cntry" in dev_dfs:
            ddf = dev_dfs["cntry"]
            row_d = ddf[ddf["group"] == c]
            if row_d.empty:
                continue
            r = row_d.iloc[0]
            mn   = float(r["mean_dev"])
            ci_lo = float(r["ci_lower_dev"])
            ci_hi = float(r["ci_upper_dev"])
            per_model = []
            for m in all_models:
                if m not in overall:
                    continue
                sub = cntry_df[
                    (cntry_df["model"] == m) & (cntry_df["group"] == c) &
                    (cntry_df["n_respondents_mean"] >= min_n_respondents)
                ]
                if sub.empty:
                    continue
                per_model.append((m, float(sub.iloc[0]["mean_alignment"]) - overall[m]))
            all_rows_meta.append(dict(
                kind="country", var="cntry", group=c,
                mean_dev=mn, ci_lo=ci_lo, ci_hi=ci_hi,
                n_models=len(per_model), per_model=per_model,
            ))
        else:
            devs, ses = [], []
            per_model = []
            for m in all_models:
                if m not in overall:
                    continue
                sub = cntry_df[
                    (cntry_df["model"] == m) & (cntry_df["group"] == c) &
                    (cntry_df["n_respondents_mean"] >= min_n_respondents)
                ]
                if sub.empty:
                    continue
                r   = sub.iloc[0]
                dev = float(r["mean_alignment"]) - overall[m]
                se  = (float(r["ci_upper"]) - float(r["ci_lower"])) / (2 * z_alpha)
                devs.append(dev)
                ses.append(se)
                per_model.append((m, dev))
            if not devs:
                continue
            n    = len(devs)
            mn   = float(np.mean(devs))
            se_c = float(np.sqrt(np.sum(np.array(ses) ** 2))) / n
            all_rows_meta.append(dict(
                kind="country", var="cntry", group=c,
                mean_dev=mn, ci_lo=mn - z_alpha * se_c, ci_hi=mn + z_alpha * se_c,
                n_models=n, per_model=per_model,
            ))

    if not all_rows_meta:
        warnings.warn("No deviation rows computed; skipping cross-model deviation plots.")
        return

    model_markers_map = {m: _MARKERS[i % len(_MARKERS)] for i, m in enumerate(all_models)}

    # ── Helper: compute y positions and var_boundaries for any row subset ─────
    def _compute_layout(rows: list[dict]):
        y_pos: list[float] = []
        current_y = 0.0
        prev_var  = None
        for row in rows:
            if row["var"] != prev_var and prev_var is not None:
                current_y += 0.4
            y_pos.append(current_y)
            current_y += 1.0
            prev_var = row["var"]
        y = np.array(y_pos)
        bounds: list[tuple[int, int, str]] = []
        cur_var, start_i = None, 0
        for idx, row in enumerate(rows):
            if row["var"] != cur_var:
                if cur_var is not None:
                    bounds.append((start_i, idx - 1, cur_var))
                cur_var  = row["var"]
                start_i  = idx
        if cur_var is not None:
            bounds.append((start_i, len(rows) - 1, cur_var))
        return y, bounds

    def _draw_deviation_fig(
        rows_meta: list[dict],
        show_ci: bool,
        show_models: bool,
        fname: str,
        title_suffix: str,
        subtitle: str = "",
    ):
        if not rows_meta:
            return
        row_y, row_bounds = _compute_layout(rows_meta)

        fig_height = max(16, len(rows_meta) * 0.7)
        fig, ax = plt.subplots(figsize=(14, fig_height))

        # Alternating backgrounds
        if all(r["var"] == "cntry" for r in rows_meta):
            # Per-country alternating bands
            for i, row in enumerate(rows_meta):
                if i % 2 == 0:
                    ax.axhspan(row_y[i] - 0.5, row_y[i] + 0.5,
                               facecolor="#F5F5F5", alpha=0.5, zorder=0)
        else:
            for i, (s_idx, e_idx, var) in enumerate(row_bounds):
                fc = "#F5F5F5" if i % 2 == 0 else "white"
                ax.axhspan(row_y[s_idx] - 0.5, row_y[e_idx] + 0.5,
                           facecolor=fc, alpha=0.5, zorder=0)

        for idx, row in enumerate(rows_meta):
            var   = row["var"]
            color = _VAR_COLORS_MASTER.get(var, "#2C3E50") if var != "cntry" else "#2C3E50"
            mn    = row["mean_dev"]
            ci_lo = row["ci_lo"]
            ci_hi = row["ci_hi"]

            # Individual model dots (variant C)
            if show_models:
                n_m = len(row["per_model"])
                offsets = (
                    np.linspace(-0.28 / 2, 0.28 / 2, n_m) if n_m > 1 else [0.0]
                )
                for (m, dev), jit in zip(row["per_model"], offsets):
                    ax.scatter(dev, row_y[idx] + jit,
                               color="#AAAAAA", marker=model_markers_map[m],
                               s=120, alpha=0.7, zorder=2, edgecolors="none")

            # CI line (variants B and C)
            if show_ci:
                ax.plot([ci_lo, ci_hi], [row_y[idx], row_y[idx]],
                        color=color, linewidth=5, alpha=0.8, zorder=3)

            # Mean dot
            ax.scatter(mn, row_y[idx], color=color, s=300, zorder=4,
                       edgecolors="white", linewidths=2.5)

        # Reference dashed line at 0
        ax.axvline(0, color="#808080", linestyle="--", linewidth=4, alpha=0.7, zorder=1)

        # y-axis labels
        y_labels = [
            _COUNTRY_NAMES.get(str(r["group"]), str(r["group"]))
            if r["var"] == "cntry" else _fmt_group(r["group"], r["var"])
            for r in rows_meta
        ]
        ax.set_yticks(row_y)
        ax.set_yticklabels(y_labels, fontsize=40)
        ax.invert_yaxis()
        y_margin = 0.3
        ax.set_ylim(row_y.max() + y_margin, row_y.min() - y_margin)

        # Right-side variable labels – always wrap at the nearest word to the midpoint
        def _wrap_lbl(s: str) -> str:
            words = s.split()
            if len(words) <= 1:
                return s
            target = len(s) / 2
            best_split, best_d, acc = 1, float("inf"), 0
            for wi, w in enumerate(words[:-1]):
                acc += len(w) + (1 if wi else 0)
                d = abs(acc - target)
                if d < best_d:
                    best_d, best_split = d, wi + 1
            return " ".join(words[:best_split]) + "\n" + " ".join(words[best_split:])

        for s_idx, e_idx, var in row_bounds:
            mid = (row_y[s_idx] + row_y[e_idx]) / 2
            lbl = "Country" if var == "cntry" else VARIABLE_LABELS.get(var, var)
            lbl = _wrap_lbl(lbl)
            color_lbl = "#2C3E50" if var == "cntry" else _VAR_COLORS_MASTER.get(var, "#2C3E50")
            ax.text(0.985, mid, lbl, va="center", ha="right", fontsize=36,
                    fontweight="bold", transform=ax.get_yaxis_transform(),
                    color=color_lbl, linespacing=1.4)

        # x-axis limits tight to data
        flat = [r["mean_dev"] for r in rows_meta]
        if show_ci:
            flat += [r["ci_lo"] for r in rows_meta]
            flat += [r["ci_hi"] for r in rows_meta]
        if show_models:
            for row in rows_meta:
                flat += [dev for _, dev in row["per_model"]]
        if flat:
            dmin    = float(np.nanmin(flat))
            dmax    = float(np.nanmax(flat))
            padding = max(0.002, (dmax - dmin) * 0.01)
            ax.set_xlim(dmin - padding, dmax + padding)

        # Legend
        if show_models:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], marker=model_markers_map[m], color="#AAAAAA",
                       markerfacecolor="#AAAAAA", markersize=14, linewidth=0, label=m)
                for m in all_models
            ]
            handles.append(Line2D([0], [0], color="#808080", linestyle="--",
                                  linewidth=4, label="Overall mean (dev = 0)"))
            ax.legend(handles=handles, loc="lower left", fontsize=20, framealpha=0.9)
        else:
            from matplotlib.lines import Line2D
            ax.legend(
                handles=[Line2D([0], [0], color="#808080", linestyle="--",
                                linewidth=4, label="Overall mean (deviation = 0)")],
                loc="lower left", fontsize=32, framealpha=0.9,
            )

        ax.set_xlabel("Alignment deviation from overall mean", fontsize=38, fontweight="bold")
        ax.tick_params(axis="x", labelsize=36)
        ax.grid(axis="x", alpha=0.3, linestyle="--", color="gray", zorder=1)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.spines["left"].set_color("#CCCCCC")

        # ax.set_title("Cross-Model Mean Deviation from Overall Alignment",
        #              fontsize=32, fontweight="bold", pad=12)

        plt.tight_layout()
        plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)
        out_path = output_dir / fname
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # Split rows into sociodemographic and country subsets
    sd_rows_meta      = [r for r in all_rows_meta if r["kind"] == "sd"]
    country_rows_meta = [r for r in all_rows_meta if r["kind"] == "country"]

    qs_label = f"  [{question_set_key}]" if question_set_key else ""

    for show_ci, show_models, suffix, ts in [
        (False, False, "means",  "means only"),
        (True,  False, "ci",     f"{int(conf_level * 100)}% CI (propagated from bootstrap)"),
        (True,  True,  "models", f"{int(conf_level * 100)}% CI + per-model dots"),
    ]:
        if suffix not in variants:
            continue
        _draw_deviation_fig(
            rows_meta=sd_rows_meta,
            show_ci=show_ci, show_models=show_models,
            fname=f"cross_model_mean_deviation_from_overall_alignment_{suffix}_sd.png",
            title_suffix=ts,
            subtitle=f"Sociodemographic groups  |  {len(all_models)} models{qs_label}  |  {ts}",
        )
        _draw_deviation_fig(
            rows_meta=country_rows_meta,
            show_ci=show_ci, show_models=show_models,
            fname=f"cross_model_mean_deviation_from_overall_alignment_{suffix}_cntry.png",
            title_suffix=ts,
            subtitle=f"Countries  |  {len(all_models)} models{qs_label}  |  {ts}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Overall alignment comparison across bootstrap methods / question sets
# ─────────────────────────────────────────────────────────────────────────────

def plot_overall_alignment_comparison(
    sources: list[tuple[str, Path]],
    output_dir: Path,
    question_set_key: str = "",
) -> None:
    """
    Forest-style plot comparing overall alignment CIs per model across bootstrap
    sources (e.g. regular bootstrap all_answered, regular bootstrap all,
    double bootstrap all_answered).

    Parameters
    ----------
    sources : list of (label, csv_path) pairs
        Each csv_path should point to an ``alignment_overall_population.csv``.
        Sources are rendered in list order, jittered vertically within each
        model row.
    output_dir : Path
    question_set_key : str

    Saved as ``overall_alignment_comparison.png``.
    """
    from matplotlib.lines import Line2D

    plt.style.use("default")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load CSVs ─────────────────────────────────────────────────────────
    loaded: list[tuple[str, pd.DataFrame]] = []
    for label, csv_path in sources:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            warnings.warn(f"Overall alignment CSV not found (skipped): {csv_path}")
            continue
        loaded.append((label, pd.read_csv(csv_path)))

    if not loaded:
        warnings.warn("No overall alignment CSVs found; skipping comparison plot.")
        return

    # ── Collect all model names, sort by first-source mean (desc) ─────────
    all_models_set: set[str] = set()
    for _, df in loaded:
        all_models_set.update(df["model"].unique())

    first_df = loaded[0][1]
    model_means: dict[str, float] = {
        row["model"]: float(row["mean_alignment"])
        for _, row in first_df.iterrows()
    }
    all_models = sorted(all_models_set, key=lambda m: model_means.get(m, 0.0), reverse=True)
    n_models = len(all_models)

    # ── Colours (colorblind-friendly, Okabe-Ito based) ────────────────────
    _SRC_COLORS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC"]
    src_colors = {label: _SRC_COLORS[i % len(_SRC_COLORS)] for i, (label, _) in enumerate(loaded)}
    src_markers = ['o', 's', '^', 'D', 'v']
    src_marker_map = {label: src_markers[i % len(src_markers)] for i, (label, _) in enumerate(loaded)}

    # ── Jitter offsets (spread ±0.18 across sources within each model row) ─
    n_src = len(loaded)
    jitter = np.linspace(-0.18, 0.18, n_src) if n_src > 1 else np.array([0.0])

    # ── Figure ────────────────────────────────────────────────────────────
    fig_height = max(6, n_models * 0.65)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    # Alternating background bands per model row
    for i in range(n_models):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, facecolor="#F5F5F5", alpha=0.5, zorder=0)

    for m_idx, model in enumerate(all_models):
        for s_idx, (label, df) in enumerate(loaded):
            sub = df[df["model"] == model]
            if sub.empty:
                continue
            row = sub.iloc[0]
            y     = m_idx + jitter[s_idx]
            color = src_colors[label]
            marker = src_marker_map[label]
            ax.plot(
                [row["ci_lower"], row["ci_upper"]], [y, y],
                color=color, linewidth=2.5, alpha=0.8, zorder=2,
            )
            ax.scatter(
                row["mean_alignment"], y,
                color=color, marker=marker, s=90, zorder=3,
                edgecolors="white", linewidths=1.5,
            )

    # ── Legend ────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D(
            [0], [0],
            color=src_colors[label],
            marker=src_marker_map[label],
            linewidth=2.5, markersize=7,
            label=label,
        )
        for label, _ in loaded
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=16, framealpha=0.9)

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(all_models, fontsize=16)
    ax.invert_yaxis()
    ax.set_ylim(n_models - 0.5, -0.5)

    all_vals = np.concatenate([
        df[["ci_lower", "ci_upper", "mean_alignment"]].values.flatten()
        for _, df in loaded
    ])
    pad = max(0.005, (np.nanmax(all_vals) - np.nanmin(all_vals)) * 0.10)
    ax.set_xlim(max(0.0, np.nanmin(all_vals) - pad), min(1.0, np.nanmax(all_vals) + pad))

    ax.set_xlabel("Overall Alignment Score", fontsize=20, fontweight="bold")
    ax.tick_params(axis="x", labelsize=16)
    qs_label = f"  [{question_set_key}]" if question_set_key else ""
    # ax.set_title(
    #     f"Overall Alignment by Model — Bootstrap Comparison{qs_label}",
    #     fontsize=17, fontweight="bold", pad=12,
    # )
    ax.grid(axis="x", alpha=0.3, linestyle="--", color="gray")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_color("#CCCCCC")

    plt.tight_layout()
    out_path = output_dir / "overall_alignment_by_model_bootstrap_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
