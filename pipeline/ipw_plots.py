"""
Plots for the IPW analysis.  Called from `pipeline.ipw.run_ipw`
(via `_write_qset_artifacts`).

All plots are saved to PNG; no GUI backend is used.

Plots:
- plot_propensity_histograms                : per-country propensity diagnostic
- plot_balance_per_variable                 : SMD pre/post by SD variable
- plot_balance_per_country                  : SMD pre/post by country
- plot_country_means_three_stage            : country forest (initial / overlap / reweighted)
- plot_cross_model_deviation_before_after   : clip-only before-after dev panel
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import COLOR_PALETTE


# ─────────────────────────────────────────────────────────────────────────────
# Propensity diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def plot_propensity_histograms(
    proba: np.ndarray,
    country_idx: np.ndarray,
    classes: list[str],
    output_path: Path,
    question_set_key: str = "",
) -> None:
    """
    Per-country histogram of P̂(C=c | X): in-country vs rest-of-sample.

    Shared x-scale across panels so countries are directly comparable.
    Country titles use full English names from ``config.COUNTRY_NAME``.
    x-axis labelled on the bottom row; y-axis on the leftmost column; one
    figure-level legend.
    """
    from .config import COUNTRY_NAME

    K = len(classes)
    # 3 cols x 10 rows for K=30: portrait layout (~6"x9.5") that fits a
    # single paper page without compressing the per-country densities.
    n_cols = min(3, K)
    n_rows = math.ceil(K / n_cols)

    # Global x-limit: the largest in-country propensity (with a small margin)
    global_max = 0.0
    for k in range(K):
        m = proba[country_idx == k, k]
        if m.size:
            global_max = max(global_max, float(m.max()))
    x_hi = max(0.05, global_max * 1.05)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols, 0.95 * n_rows),
        sharex=True, sharey=False,
    )
    axes = np.atleast_2d(axes).flatten()
    bins = np.linspace(0, x_hi, 40)

    legend_handles = None
    for k, ax in enumerate(axes):
        if k >= K:
            ax.set_visible(False)
            continue
        c = classes[k]
        in_c  = proba[country_idx == k, k]
        out_c = proba[country_idx != k, k]
        if len(in_c) == 0:
            ax.set_visible(False)
            continue
        h_out = ax.hist(out_c, bins=bins, color="#cccccc",
                        alpha=0.75, density=True, label="rest of sample")
        h_in  = ax.hist(in_c,  bins=bins, color=COLOR_PALETTE[0],
                        alpha=0.75, density=True, label="in country")
        if legend_handles is None:
            legend_handles = [h_in[2][0], h_out[2][0]]
        name = COUNTRY_NAME.get(c, c)
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, x_hi)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=8)

        # Label axes only on edges
        row, col = divmod(k, n_cols)
        if row != n_rows - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Predicted P(C = c | X)", fontsize=9)
        if col == 0:
            ax.set_ylabel("Density", fontsize=9)

    if legend_handles is not None:
        fig.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, color=COLOR_PALETTE[0], alpha=0.75),
                plt.Rectangle((0, 0), 1, 1, color="#cccccc", alpha=0.75),
            ],
            labels=["in country (target)", "rest of sample (control)"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=2,
            fontsize=10,
            framealpha=0.9,
        )

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Balance bar charts
# ─────────────────────────────────────────────────────────────────────────────

def _balance_bar(
    df: pd.DataFrame,
    label_col: str,
    output_path: Path,
    title: str,
    xlabel: str = "",
    rotation: int = 0,
    horizontal: bool = False,
) -> None:
    """
    Shared internal: paired pre/post SMD bars from ``df`` with columns
    ``mean_smd_pre`` and ``mean_smd_post`` plus a ``label_col`` for labels.
    Threshold lines at 0.10 and 0.20.
    """
    if df.empty:
        return
    labels = df[label_col].astype(str).tolist()
    pre    = df["mean_smd_pre"].values
    post   = df["mean_smd_post"].values
    n = len(labels)

    if horizontal:
        # Per-variable bar chart: variables on y, SMD on x
        fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.35 * n + 1.0)))
        y = np.arange(n)
        h = 0.38
        ax.barh(y + h/2, pre,  height=h, color="#bdbdbd", label="pre-IPW")
        ax.barh(y - h/2, post, height=h, color=COLOR_PALETTE[0], label="post-IPW")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        for thr, lab, col in [(0.10, "good", "tab:green"), (0.20, "imbalanced", "tab:red")]:
            ax.axvline(thr, ls="--", color=col, lw=0.9, alpha=0.7)
            ax.text(thr, n - 0.5, f"  {lab} ({thr})",
                    color=col, fontsize=8, va="top")
        ax.set_xlabel(xlabel or "Mean |SMD|", fontsize=10)
    else:
        # Per-country bar chart: countries on x, SMD on y
        fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * n + 1.5), 5.0))
        x = np.arange(n)
        w = 0.38
        ax.bar(x - w/2, pre,  width=w, color="#bdbdbd", label="pre-IPW")
        ax.bar(x + w/2, post, width=w, color=COLOR_PALETTE[0], label="post-IPW")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center",
                           fontsize=9)
        for thr, lab, col in [(0.10, "good", "tab:green"), (0.20, "imbalanced", "tab:red")]:
            ax.axhline(thr, ls="--", color=col, lw=0.9, alpha=0.7)
            ax.text(n - 0.5, thr, f"{lab} ({thr})  ",
                    color=col, fontsize=8, ha="right", va="bottom")
        ax.set_ylabel(xlabel or "Mean |SMD|", fontsize=10)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x" if horizontal else "y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_balance_per_variable(
    bal_var: pd.DataFrame,
    output_path: Path,
    question_set_key: str = "",
) -> None:
    """
    Per SD variable, paired pre/post mean SMD bars.

    Expects the DataFrame from ``ipw.balance_per_variable``: variable,
    mean_smd_pre, mean_smd_post, improvement, …  Sorted by descending
    post-IPW SMD for visual scan.  Raw variable names are rendered with
    the human-readable labels from ``config.VARIABLE_LABELS``.
    """
    from .config import VARIABLE_LABELS

    df = bal_var.sort_values("mean_smd_post", ascending=False).reset_index(drop=True)
    df = df.copy()
    df["variable_label"] = df["variable"].map(lambda v: VARIABLE_LABELS.get(v, v))
    qs = f"  [{question_set_key}]" if question_set_key else ""
    _balance_bar(
        df, label_col="variable_label",
        output_path=output_path,
        title=f"Balance per SD variable (clip){qs}",
        xlabel="Mean |SMD| across 30 countries",
        horizontal=True,
    )


def plot_balance_per_country(
    bal_country: pd.DataFrame,
    output_path: Path,
    question_set_key: str = "",
) -> None:
    """
    Per country, paired pre/post mean SMD bars.

    Expects the DataFrame from ``ipw.balance_per_country``: country,
    country_name, mean_smd_pre, mean_smd_post, …  Uses country_name on the
    x-axis.  Sorted by descending post-IPW SMD.
    """
    df = bal_country.sort_values("mean_smd_post", ascending=False).reset_index(drop=True)
    qs = f"  [{question_set_key}]" if question_set_key else ""
    _balance_bar(
        df, label_col="country_name",
        output_path=output_path,
        title=f"Balance per country (clip){qs}",
        xlabel="Mean |SMD| across 14 SD variables",
        rotation=55,
        horizontal=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Two-stage country forest (main paper figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_country_means_three_stage(
    df_country: pd.DataFrame,
    country_mean_ci: pd.DataFrame | None,
    df: pd.DataFrame,
    country_idx: np.ndarray,
    classes: list[str],
    dev: np.ndarray,
    pspwght: np.ndarray,
    fw_primary: np.ndarray,
    output_path: Path,
    question_set_key: str = "",
    pspwght_survivors_means: pd.DataFrame | None = None,
    pspwght_survivors_ci: pd.DataFrame | None = None,
) -> None:
    """
    Forest plot of country-level cross-model mean deviation, up to three
    rows per country:

    initial            : pspwght-weighted mean on full sample (always shown)
    overlap_no_reweight: pspwght-weighted mean on the surviving subset
                         (drop settings only — passed via
                         ``pspwght_survivors_means``)
    reweighted         : IPW-weighted mean on the surviving subset

    Bootstrap CIs are drawn from ``country_mean_ci`` (for the reweighted row)
    and ``pspwght_survivors_ci`` (for the overlap-no-reweight row), when
    provided.  Countries are sorted by their initial mean.
    """
    K = len(classes)
    means_initial    = np.full(K, np.nan)
    means_reweighted = np.full(K, np.nan)
    for k in range(K):
        m_pre = (country_idx == k) & np.isfinite(dev) & (pspwght > 0)
        m_pst = (country_idx == k) & np.isfinite(dev) & (fw_primary > 0)
        if m_pre.any():
            means_initial[k] = float(np.average(dev[m_pre], weights=pspwght[m_pre]))
        if m_pst.any():
            means_reweighted[k] = float(np.average(dev[m_pst], weights=fw_primary[m_pst]))

    # Optional overlap-no-reweight (drop-setting only): map per-country
    means_overlap = np.full(K, np.nan)
    has_overlap = (
        pspwght_survivors_means is not None
        and not pspwght_survivors_means.empty
        and "country" in pspwght_survivors_means.columns
        and "mean"    in pspwght_survivors_means.columns
    )
    if has_overlap:
        ov = pspwght_survivors_means.set_index("country")
        for k, c in enumerate(classes):
            if c in ov.index:
                means_overlap[k] = float(ov.loc[c, "mean"])

    from .config import COUNTRY_NAME

    order = np.argsort(np.where(np.isnan(means_initial), -np.inf, means_initial))
    classes_sorted = [classes[i] for i in order]
    labels_sorted  = [COUNTRY_NAME.get(c, c) for c in classes_sorted]

    fig, ax = plt.subplots(figsize=(8, max(8, 0.32 * K)))
    y_base  = np.arange(K) * 1.0
    if has_overlap:
        offsets = {"initial": -0.22, "overlap_no_reweight": 0.0, "reweighted": +0.22}
    else:
        offsets = {"initial": -0.15, "reweighted": +0.15}
    colors  = {"initial":             "#888888",
               "overlap_no_reweight": COLOR_PALETTE[3],   # orange
               "reweighted":          COLOR_PALETTE[0]}   # brick red
    # Labels — the "drop-survivors" qualifier only applies to drop settings.
    if has_overlap:
        labels  = {"initial":             "Initial (pspwght, full sample)",
                   "overlap_no_reweight": "Overlap (pspwght, drop-survivors only)",
                   "reweighted":          "Reweighted (IPW × pspwght, drop-survivors)"}
    else:
        labels  = {"initial":             "Initial (pspwght)",
                   "reweighted":          "Reweighted (IPW × pspwght)"}
    means_map = {"initial":             means_initial,
                 "overlap_no_reweight": means_overlap,
                 "reweighted":          means_reweighted}

    def _ci_arr(ci_df):
        lo = np.full(K, np.nan); hi = np.full(K, np.nan)
        if ci_df is not None and not ci_df.empty:
            cmci = ci_df.set_index("country")
            for k, c in enumerate(classes):
                if c in cmci.index:
                    lo[k] = float(cmci.loc[c, "ci_lo"])
                    hi[k] = float(cmci.loc[c, "ci_hi"])
        return lo, hi

    ci_lo_rew,    ci_hi_rew    = _ci_arr(country_mean_ci)
    ci_lo_overl,  ci_hi_overl  = _ci_arr(pspwght_survivors_ci)

    for stage, off in offsets.items():
        m_ord = means_map[stage][order]
        x_err = None
        if stage == "reweighted" and not np.all(np.isnan(ci_lo_rew)):
            lo = ci_lo_rew[order]; hi = ci_hi_rew[order]
            x_err = np.vstack([np.maximum(m_ord - lo, 0.0),
                               np.maximum(hi - m_ord, 0.0)])
        elif stage == "overlap_no_reweight" and not np.all(np.isnan(ci_lo_overl)):
            lo = ci_lo_overl[order]; hi = ci_hi_overl[order]
            x_err = np.vstack([np.maximum(m_ord - lo, 0.0),
                               np.maximum(hi - m_ord, 0.0)])
        ax.errorbar(
            m_ord, y_base + off, xerr=x_err,
            fmt="o", color=colors[stage], markersize=4,
            ecolor=colors[stage], alpha=0.85, linewidth=0,
            elinewidth=1.2, capsize=2, label=labels[stage],
        )

    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_yticks(y_base)
    ax.set_yticklabels(labels_sorted, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-model mean deviation\n(positive = above European pool mean)",
                  fontsize=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-model mean deviation: "before vs after" (clip-only, no CIs)
# Inspired by pipeline.plots.plot_cross_model_mean_deviation variant C.
# ─────────────────────────────────────────────────────────────────────────────

_MAIN_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']


def plot_cross_model_deviation_before_after(
    classes: list[str],
    per_country_cross_model_dev_pre: pd.DataFrame,
    per_country_cross_model_dev_post: pd.DataFrame,
    per_llm_country_dev_pre: dict[str, dict[str, float]],
    per_llm_country_dev_post: dict[str, dict[str, float]],
    output_path: Path,
    question_set_key: str = "",
) -> None:
    """
    Per-country cross-model alignment deviation, pre vs post clip-IPW.

    Each row, for one country:
      * background: 10 per-LLM "after" markers (LLM-specific shape) joined by
        a thin grey line back to that LLM's "before" position — neutral grey
        for all LLM lines/markers.
      * foreground: cross-model "after" marker joined to its "before" marker
        by a coloured line whose hue encodes the **direction relative to
        zero**:
          * |post| < |pre|  → "toward zero" (red)  — IPW pulls the country
            mean towards the European pool average.
          * |post| > |pre|  → "outward"     (blue) — IPW accentuates the
            country's distance from the average.
          * |post| ≈ |pre|  → "no change"   (grey).

    Y-axis ordering matches the main-analysis ``plot_cross_model_mean_deviation``
    convention: countries sorted ascending by cross-model mean alignment under
    pspwght (the "before" position).
    """
    from .config import COUNTRY_NAME

    pre  = per_country_cross_model_dev_pre.set_index("country")["mean"]
    post = per_country_cross_model_dev_post.set_index("country")["mean"]

    countries_present = [c for c in classes if c in pre.index and c in post.index]
    if not countries_present:
        return
    countries_sorted = sorted(countries_present, key=lambda c: float(pre.loc[c]))

    all_llms = sorted({m for d in per_llm_country_dev_post.values() for m in d})
    llm_markers = {m: _MAIN_MARKERS[i % len(_MAIN_MARKERS)]
                   for i, m in enumerate(all_llms)}

    # Figure dimensions / typography copied from the main-analysis
    # `plot_cross_model_mean_deviation` country panel so the IPW figure
    # reads at the same scale as the rest of the paper.
    n = len(countries_sorted)
    fig_height = max(16.0, n * 0.7)
    fig, ax = plt.subplots(figsize=(14.0, fig_height))
    y_pos = np.arange(n)

    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(y_pos[i] - 0.5, y_pos[i] + 0.5,
                       facecolor="#F5F5F5", alpha=0.5, zorder=0)

    TOWARD_COLOR  = "#8B1A1A"   # darker brick red — toward zero
    OUTWARD_COLOR = "#0A5B7C"   # dark teal — outward
    NEUTRAL_COLOR = "#9CA3AF"   # grey — essentially no change
    NEUTRAL_TOL   = 1e-5

    pre_marker_color  = "#9CA3AF"
    llm_pre_color     = "#D5D8DC"
    llm_post_color    = "#9CA3AF"

    def _direction_color(d_pre: float, d_post: float) -> str:
        if not (np.isfinite(d_pre) and np.isfinite(d_post)):
            return NEUTRAL_COLOR
        gap = abs(d_post) - abs(d_pre)
        if gap < -NEUTRAL_TOL:
            return TOWARD_COLOR
        if gap >  NEUTRAL_TOL:
            return OUTWARD_COLOR
        return NEUTRAL_COLOR

    n_llm = max(len(all_llms), 1)
    jitter_offsets = (np.linspace(-0.32, 0.32, n_llm)
                      if n_llm > 1 else np.array([0.0]))
    llm_jitter = {m: jitter_offsets[i] for i, m in enumerate(all_llms)}

    for idx, c in enumerate(countries_sorted):
        y = y_pos[idx]
        pre_llm  = per_llm_country_dev_pre.get(c, {})
        post_llm = per_llm_country_dev_post.get(c, {})
        for m in all_llms:
            d_pre  = pre_llm.get(m)
            d_post = post_llm.get(m)
            if d_pre is None or d_post is None:
                continue
            if not (np.isfinite(d_pre) and np.isfinite(d_post)):
                continue
            yj = y + llm_jitter[m]
            ax.plot([d_pre, d_post], [yj, yj],
                    color=llm_post_color, linewidth=1.8, alpha=0.55, zorder=2)
            ax.scatter(d_pre,  yj, marker=llm_markers[m],
                       facecolor="none", edgecolor=llm_pre_color,
                       s=110, linewidths=1.6, alpha=0.9, zorder=2)
            ax.scatter(d_post, yj, marker=llm_markers[m],
                       facecolor=llm_post_color, edgecolor="none",
                       s=140, alpha=0.9, zorder=3)

        pre_val  = float(pre.loc[c])
        post_val = float(post.loc[c])
        cross_color = _direction_color(pre_val, post_val)
        ax.plot([pre_val, post_val], [y, y],
                color=cross_color, linewidth=5.0, alpha=0.9, zorder=4)
        ax.scatter(pre_val, y, marker="o", facecolor="white",
                   edgecolor=pre_marker_color, s=260, linewidths=3.0, zorder=5)
        ax.scatter(post_val, y, marker="o", facecolor=cross_color,
                   edgecolor="white", s=320, linewidths=3.0, zorder=6)

    ax.axvline(0, color="#808080", linestyle="--", linewidth=4.0, alpha=0.7, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([COUNTRY_NAME.get(c, c) for c in countries_sorted], fontsize=40)
    ax.invert_yaxis()
    y_margin = 0.3
    ax.set_ylim(y_pos.max() + y_margin, y_pos.min() - y_margin)

    flat = []
    for c in countries_sorted:
        flat.append(float(pre.loc[c]))
        flat.append(float(post.loc[c]))
        for m, v in per_llm_country_dev_pre.get(c, {}).items():
            if np.isfinite(v):
                flat.append(float(v))
        for m, v in per_llm_country_dev_post.get(c, {}).items():
            if np.isfinite(v):
                flat.append(float(v))
    dmin = float(np.nanmin(flat))
    dmax = float(np.nanmax(flat))
    pad  = max(0.002, (dmax - dmin) * 0.04)
    ax.set_xlim(dmin - pad, dmax + pad)

    from matplotlib.lines import Line2D
    cross_handles = [
        Line2D([0], [0], marker="o", markerfacecolor="white",
               markeredgecolor=pre_marker_color, markersize=18,
               linewidth=0, label="Cross-model pre"),
        Line2D([0], [0], color=TOWARD_COLOR, linewidth=5.0,
               label="Toward zero ($|post|<|pre|$)"),
        Line2D([0], [0], color=OUTWARD_COLOR, linewidth=5.0,
               label="Outward ($|post|>|pre|$)"),
        Line2D([0], [0], color="#808080", linestyle="--",
               linewidth=4.0, label="Overall mean (dev = 0)"),
    ]
    llm_handles = [
        Line2D([0], [0], marker=llm_markers[m], color=llm_post_color,
               markerfacecolor=llm_post_color, markersize=14, linewidth=0,
               label=m)
        for m in all_llms
    ]

    ax.set_xlabel("Alignment deviation from overall mean",
                  fontsize=38, fontweight="bold")
    ax.tick_params(axis="x", labelsize=36)
    ax.grid(axis="x", alpha=0.3, linestyle="--", color="gray", zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_color("#CCCCCC")

    # LLM markers legend pinned to bottom-left; cross-model legend stacked
    # directly above it.  bbox_transform=ax.transAxes anchors via axes
    # fraction; we measure the LLM legend after the first draw to place the
    # cross-model legend so they don't overlap regardless of marker count.
    leg_llm = ax.legend(
        handles=llm_handles,
        loc="lower left", bbox_to_anchor=(0.01, 0.005),
        bbox_transform=ax.transAxes,
        fontsize=20, framealpha=0.9, ncol=2,
        title="LLM markers", title_fontsize=20,
        borderpad=0.6, labelspacing=0.4, columnspacing=1.2,
    )
    ax.add_artist(leg_llm)
    fig.canvas.draw()
    llm_bbox = leg_llm.get_window_extent().transformed(ax.transAxes.inverted())
    cross_anchor_y = llm_bbox.y1 + 0.015   # small gap above LLM legend
    ax.legend(
        handles=cross_handles,
        loc="lower left", bbox_to_anchor=(0.01, cross_anchor_y),
        bbox_transform=ax.transAxes,
        fontsize=20, framealpha=0.9, ncol=1,
        title="Direction & reference", title_fontsize=20,
        borderpad=0.6, labelspacing=0.4,
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
