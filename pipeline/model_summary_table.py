"""
model_summary_table.py – Per-model summary statistics table (CSV + LaTeX).

Columns : one per model in MODEL_REGISTRY (short alias).

Q is the full 'all' question set; Q~ is the 'all_answered' subset, i.e. the
questions every model answered.

Rows — answer statistics
------------------------
  1.  Questions answered on Q                   n
  2.  Questions answered on Q                   % of the 'all' set
  3.  Uniform answers on Q                      % of answered questions
        (all seeds gave the same answer)
  4.  Mean disagreement from the majority vote  % of answered questions
  5.  Mean number of answers differing from MV  count, on Q
  6.  Refusal rate on Q                         %
  7.  Refusal rate on Q~                        %
  8.  Questions where all seeds refused, on Q   n
  9.  Mean number of answers differing from MV  count, on Q~

Rows — alignment scores
-----------------------
 10.  Population alignment [all]                mean
 11.  Population alignment [all]                95% CI
 12.  Population alignment [all_answered]       mean
 13.  Population alignment [all_answered]       95% CI

Alignment rows are read from the bootstrap output under *results_dir*; they
show as '—' when the corresponding bootstrap has not been run.

Output
------
  model_summary_table.csv
  model_summary_table.tex

Entry point
-----------
    run_model_summary_table(
        votes_csvs, refusal_stats_csv, questions_csv,
        results_dir, output_dir,
    )
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    MODEL_REGISTRY,
    RESULTS_DIR,
)


# ── Constants ─────────────────────────────────────────────────────────────────

_BOOT_COLS = [
    "variable", "group", "model",
    "mean_alignment", "median_alignment", "std_alignment",
    "ci_lower", "ci_upper", "ci_width",
    "n_bootstrap", "n_respondents_mean",
]

_ROW_LABELS = [
    # Answer statistics (9 rows)
    "Questions answered (n)",
    "Questions answered (%)",
    "Uniform answers Q (%)",
    "Mean disagreement Q (%)",
    "Mean n answers diff. from MV",
    "Refusal rate Q (%)",
    "Refusal rate Q~ (%)",
    "All questions refused Q (n)",
    "Mean n answers diff. from MV Q~",
    # Alignment scores (4 rows)
    "Population alignment [all] mean",
    "Population alignment [all] 95% CI",
    "Population alignment [all_answered] mean",
    "Population alignment [all_answered] 95% CI",
]

# LaTeX row labels (with \hspace indentation; section headers injected by
# _write_latex).  Must stay index-aligned with _ROW_LABELS — the writer zips
# the two lists together.
_ROW_LABELS_LATEX = [
    # Answer statistics
    r"\hspace{6pt} Quest.\ ans.\ (n)",
    r"\hspace{6pt} Quest.\ ans.\ (\%)",
    r"\hspace{6pt} Unif.\ answers (\%)",
    r"\hspace{6pt} Disagreement (\%)",
    r"\hspace{6pt} Mean \# diff.\ from MV",
    r"\hspace{6pt} Refusal rate (\%)",
    r"\hspace{6pt} Refusal rate $\tilde{\mathcal{Q}}$ (\%)",
    r"\hspace{6pt} All refused",
    r"\hspace{6pt} Mean \# diff.\ from MV ($\tilde{\mathcal{Q}}$)",
    # Alignment
    r"\hspace{6pt} Overall ($A_{\mathcal{P},m,\mathcal{Q}}$)",
    r"\hspace{6pt} 95\% CI$^\dagger$",
    r"\hspace{6pt} Overall ($A_{\mathcal{P},m,\tilde{\mathcal{Q}}}$)",
    r"\hspace{6pt} 95\% CI$^\dagger$",
]

# Maps row index → section header inserted before that row
_SECTION_HEADERS = {
    0: r"\textbf{Answer Statistics on $\mathcal{Q}$}",
    9: r"\textbf{Alignment Scores}",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_bootstrap_overall(path: Path) -> pd.DataFrame:
    """Load alignment_overall_population.csv, handling with/without header."""
    if not path.exists():
        return pd.DataFrame(columns=_BOOT_COLS)
    with open(path) as fh:
        first_field = fh.readline().strip().split(",")[0]
    if first_field == "variable":
        return pd.read_csv(path)
    return pd.read_csv(path, header=None, names=_BOOT_COLS)


def _parse_answers(s) -> list[float]:
    """Parse the all_answers string column (e.g. '[4.0, 3.0, ...]') to a list."""
    if pd.isna(s):
        return []
    try:
        return [float(x) for x in ast.literal_eval(str(s))]
    except Exception:
        return []


def _fmt(val, decimals: int = 1) -> str:
    """Format a float without % sign, returning '—' for NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "\u2014"
    return f"{val:.{decimals}f}"


def _fmt_n_pct(n: int, pct: float) -> str:
    if n is None:
        return "\u2014"
    return f"{n} ({pct:.1f}%)"


def _fmt_ci_pm(lo: float, hi: float) -> str:
    r"""Format bootstrap CI as $\pm$X – half-width, 4dp, no leading zero."""
    if lo is None or hi is None or (isinstance(lo, float) and np.isnan(lo)):
        return "\u2014"
    half = (hi - lo) / 2
    s = f"{half:.4f}"
    return f"$\\pm${s[1:] if s.startswith('0') else s}"


# ── Core computation ──────────────────────────────────────────────────────────

def _compute_metrics(
    short_alias: str,
    df_votes: pd.DataFrame,
    df_refusal: pd.DataFrame,
    df_boot_all: pd.DataFrame,
    df_boot_aa: pd.DataFrame,
    all_questions_n: int,
    qtilde_questions: set,
) -> list[str]:
    """Return a list of formatted metric strings (one per _ROW_LABELS entry)."""

    votes_m   = df_votes[df_votes["short"] == short_alias]
    refusal_m = df_refusal[df_refusal["short"] == short_alias]

    if refusal_m.empty and votes_m.empty:
        return ["\u2014"] * len(_ROW_LABELS)

    # ── 1. Questions with ≥1 valid response (on Q) ────────────────────────────
    if not refusal_m.empty:
        n_answered    = int((refusal_m["n_valid_answers"] > 0).sum())
        n_refused_all = int((refusal_m["n_valid_answers"] == 0).sum())
    else:
        n_answered    = int(len(votes_m))
        n_refused_all = max(0, all_questions_n - n_answered)
    pct_answered = n_answered / all_questions_n * 100 if all_questions_n else np.nan

    # ── Q metrics: answered questions only ───────────────────────────────────
    answered_v = votes_m[votes_m["n_valid_answers"] > 0].copy() if not votes_m.empty else pd.DataFrame()
    n_ans = len(answered_v)

    if n_ans > 0:
        pct_uniform_q = (answered_v["answer_variation"] == 1).sum() / n_ans * 100
        mean_disagr_q = answered_v["pct_disagreement"].mean()
        mean_n_diff_q = (
            answered_v["pct_disagreement"] / 100.0 * answered_v["n_valid_answers"]
        ).mean()
    else:
        pct_uniform_q = np.nan
        mean_disagr_q = np.nan
        mean_n_diff_q = np.nan

    # ── Refusal rate on Q (all questions) ─────────────────────────────────────
    if not refusal_m.empty:
        pct_refusal_q = (
            refusal_m["n_refusals"] / refusal_m["n_total_responses"]
        ).mean() * 100
    elif not votes_m.empty:
        n_zero = max(0, all_questions_n - len(votes_m))
        rr_vals = list((votes_m["n_valid_answers"] / votes_m["n_seeds"]).fillna(0))
        rr_vals += [0.0] * n_zero
        pct_refusal_q = (1.0 - float(np.mean(rr_vals))) * 100
    else:
        pct_refusal_q = np.nan

    # ── Q~ metrics ────────────────────────────────────────────────────────────
    if qtilde_questions:
        qt_v = votes_m[votes_m["question_var"].isin(qtilde_questions)].copy()
        qt_r = (
            refusal_m[refusal_m["question_var"].isin(qtilde_questions)].copy()
            if not refusal_m.empty else pd.DataFrame()
        )
        qt_ans = qt_v[qt_v["n_valid_answers"] > 0]
        n_qt   = len(qt_ans)

        if n_qt > 0:
            pct_uniform_qt = (qt_ans["answer_variation"] == 1).sum() / n_qt * 100
            mean_disagr_qt = qt_ans["pct_disagreement"].mean()
            mean_n_diff_qt = (
                qt_ans["pct_disagreement"] / 100.0 * qt_ans["n_valid_answers"]
            ).mean()
        else:
            pct_uniform_qt = np.nan
            mean_disagr_qt = np.nan
            mean_n_diff_qt = np.nan

        if not qt_r.empty:
            pct_refusal_qt = (
                qt_r["n_refusals"] / qt_r["n_total_responses"]
            ).mean() * 100
        elif n_qt > 0:
            pct_refusal_qt = (
                1.0 - (qt_ans["n_valid_answers"] / qt_ans["n_seeds"]).mean()
            ) * 100
        else:
            pct_refusal_qt = np.nan
            mean_n_diff_qt = np.nan
    else:
        pct_uniform_qt = np.nan
        mean_disagr_qt = np.nan
        pct_refusal_qt = np.nan
        mean_n_diff_qt = np.nan

    # ── Bootstrap alignment [all / Q] ─────────────────────────────────────────
    if not df_boot_all.empty and short_alias in df_boot_all["model"].values:
        r = df_boot_all[df_boot_all["model"] == short_alias].iloc[0]
        align_all_mean = f"{r['mean_alignment']:.3f}"
        align_all_ci   = _fmt_ci_pm(r["ci_lower"], r["ci_upper"])
    else:
        align_all_mean = "\u2014"
        align_all_ci   = "\u2014"

    # ── Bootstrap alignment [all_answered / Q~] ───────────────────────────────
    if not df_boot_aa.empty and short_alias in df_boot_aa["model"].values:
        r2 = df_boot_aa[df_boot_aa["model"] == short_alias].iloc[0]
        align_aa_mean = f"{r2['mean_alignment']:.3f}"
        align_aa_ci   = _fmt_ci_pm(r2["ci_lower"], r2["ci_upper"])
    else:
        align_aa_mean = "\u2014"
        align_aa_ci   = "\u2014"

    return [
        str(n_answered),                         # 1  Q: answered n
        _fmt(pct_answered),                      # 2  Q: answered %
        _fmt(pct_uniform_q),                     # 3  Q: uniform
        _fmt(mean_disagr_q, decimals=2),         # 4  Q: mean disagreement %
        _fmt(mean_n_diff_q, decimals=2),         # 5  Q: mean n diff from MV
        _fmt(pct_refusal_q),                     # 6  Q: refusal rate
        _fmt(pct_refusal_qt),                    # 7  Q~: refusal rate
        str(n_refused_all),                      # 8  Q: all seeds refused
        _fmt(mean_n_diff_qt, decimals=2),        # 9  Q~: mean n diff from MV
        align_all_mean,                          # 10 alignment Q
        align_all_ci,                            # 11 CI Q
        align_aa_mean,                           # 12 alignment Q~
        align_aa_ci,                             # 13 CI Q~
    ]


# ── LaTeX writer ──────────────────────────────────────────────────────────────

def _write_latex(df_table: pd.DataFrame, out_path: Path) -> None:
    """Write a styled LaTeX table with M1–M10 short headers and a bottom legend."""
    short_aliases = list(df_table.columns)
    n_cols = len(short_aliases)
    n_total = n_cols + 1  # including the label column

    # M1–M10 column labels
    m_labels = [f"M{i + 1}" for i in range(n_cols)]

    # Column spec: groups of 5 for visual clarity
    grp = n_cols // 5
    col_spec = "l " + " ".join(["ccccc"] * grp) if n_cols % 5 == 0 else "l" + "c" * n_cols

    def _escape(v: str) -> str:
        return v.replace("%", r"\%") if "%" in v and r"\%" not in v else v

    lines = [
        r"\begin{table*}[t!]",
        r"\centering",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline",
        " & " + " & ".join(f"\\textbf{{{m}}}" for m in m_labels) + r" \\",
        r"\hline",
    ]

    for i, (row_label_plain, row_label_tex) in enumerate(
        zip(_ROW_LABELS, _ROW_LABELS_LATEX)
    ):
        if i in _SECTION_HEADERS:
            if i > 0:
                lines.append(r"\hline")
            header = _SECTION_HEADERS[i]
            lines.append(
                f"\\multicolumn{{{n_total}}}{{l}}{{{header}}}" + r" \\"
            )
        # Wrap CI cell values in \small
        ci_row_indices = {i for i, lbl in enumerate(_ROW_LABELS) if "95%" in lbl}
        if i in ci_row_indices:
            row_vals = [r"\small " + _escape(str(df_table.loc[row_label_plain, col]))
                        for col in short_aliases]
        else:
            row_vals = [_escape(str(df_table.loc[row_label_plain, col]))
                        for col in short_aliases]
        lines.append(row_label_tex + " & " + " & ".join(row_vals) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    # ── Model legend (outside tabular) ───────────────────────────────────────
    legend_entries = [
        f"M{i+1}: {short_aliases[i].replace('_', r'\_')}"
        for i in range(n_cols)
    ]
    lines.append(
        r"\par\smallskip"
        r"\footnotesize\textbf{Model legend:} "
        + ",\\ ".join(legend_entries)
    )
    lines.append(
        r"\par\smallskip"
        r"\footnotesize $^\dagger$All confidence intervals are $\pm 0.005$ or less. "
        r"Full intervals provided in Appendix."
    )

    lines += [
        r"\caption{Model response statistics summary and overall alignment scores "
        r"across the whole population $\mathcal{P}$ and question set $\mathcal{Q}$, "
        r"with $95\%$ bootstrap confidence intervals. "
        r"$\tilde{\mathcal{Q}} \subset \mathcal{Q}$ denotes the subset of questions "
        r"answered by all models.}",
        r"\label{tab:model_summary}",
        r"\end{table*}",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Public entry point ────────────────────────────────────────────────────────

def run_model_summary_table(
    votes_csvs: list[Path],
    refusal_stats_csv: Path,
    questions_csv: Path,
    results_dir: Path,
    output_dir: Path,
    model_registry: dict[str, str] = MODEL_REGISTRY,
) -> None:
    """
    Build the per-model summary table and write CSV + LaTeX to *output_dir*.

    Parameters
    ----------
    votes_csvs        : list of paths to model_majority_votes.csv
    refusal_stats_csv : path to model_refusal_stats.csv
    questions_csv     : path to value_laden_Qs.csv (for 'all' set size)
    results_dir       : RESULTS_DIR (contains all/ and all_answered/ subdirs)
    output_dir        : directory to write output files into
    model_registry    : full_model_name → short_alias
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("MODEL SUMMARY TABLE")
    print("=" * 60)

    # ── Load votes ────────────────────────────────────────────────────────────
    dfs_v = []
    for p in votes_csvs:
        p = Path(p)
        if p.exists():
            dfs_v.append(pd.read_csv(p))
    if not dfs_v:
        print("  WARNING: no majority_votes CSVs found – skipping summary table.")
        return
    df_votes = pd.concat(dfs_v, ignore_index=True)
    df_votes = df_votes[df_votes["model"].isin(model_registry)].copy()
    df_votes["short"] = df_votes["model"].map(model_registry)

    # ── Load refusal stats ────────────────────────────────────────────────────
    refusal_path = Path(refusal_stats_csv)
    if refusal_path.exists():
        df_refusal = pd.read_csv(refusal_path)
        df_refusal = df_refusal[df_refusal["model"].isin(model_registry)].copy()
        df_refusal["short"] = df_refusal["model"].map(model_registry)
    else:
        print(f"  WARNING: {refusal_path} not found; response-rate metrics will use votes CSV.")
        df_refusal = pd.DataFrame()

    # ── Total 'all' question set size ─────────────────────────────────────────
    try:
        df_q = pd.read_csv(questions_csv)
        df_q["has_question"] = (
            df_q["has_question"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
        )
        all_q_n = int(df_q["has_question"].sum())
    except Exception:
        all_q_n = df_votes["question_var"].nunique()
        print(f"  WARNING: could not load {questions_csv}; using {all_q_n} as all-set size.")

    # ── Load bootstrap results ────────────────────────────────────────────────
    results_dir = Path(results_dir)
    df_boot_all = _load_bootstrap_overall(
        results_dir / "all" / "bootstrap_results" / "alignment_overall_population.csv"
    )
    df_boot_aa  = _load_bootstrap_overall(
        results_dir / "all_answered" / "bootstrap_results" / "alignment_overall_population.csv"
    )

    print(f"  All-set questions:     {all_q_n}")
    print(f"  Bootstrap [all]:       {len(df_boot_all)} model rows loaded")
    print(f"  Bootstrap [all_ans'd]: {len(df_boot_aa)} model rows loaded")

    # ── Compute Q~ = questions answered by ALL models ────────────────────────
    short_aliases = list(dict.fromkeys(model_registry.values()))  # registry order
    ref_or_votes  = df_refusal if not df_refusal.empty else df_votes
    model_answered: dict[str, set] = {}
    for alias in short_aliases:
        sub = ref_or_votes[ref_or_votes["short"] == alias]
        qs  = set(sub[sub["n_valid_answers"] > 0]["question_var"].tolist())
        if qs:
            model_answered[alias] = qs
    qtilde_questions = (
        set.intersection(*model_answered.values()) if model_answered else set()
    )
    print(f"  Q~ (intersection):     {len(qtilde_questions)} questions")

    # ── Build table ───────────────────────────────────────────────────────────
    data: dict[str, list] = {ms: [] for ms in short_aliases}
    for ms in short_aliases:
        metrics = _compute_metrics(
            ms, df_votes, df_refusal,
            df_boot_all, df_boot_aa,
            all_q_n,
            qtilde_questions,
        )
        data[ms] = metrics

    df_table = pd.DataFrame(data, index=_ROW_LABELS)
    df_table.index.name = "Metric"

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = output_dir / "model_summary_table.csv"
    df_table.to_csv(csv_path)
    print(f"\n  Saved: {csv_path}")

    # ── Write LaTeX ───────────────────────────────────────────────────────────
    tex_path = output_dir / "model_summary_table.tex"
    _write_latex(df_table, tex_path)
    print(f"  Saved: {tex_path}")
