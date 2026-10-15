"""
Method comparison for alignment score prediction (weighted, cross-validated).

Models compared
---------------
1. Country-only baseline       – LinearRegression (dummy-coded country)
2. SD-only                     – LinearRegression
3. SD-only                     – XGBoost (randomised HP search, native cats)
4. Country + SD                – LinearRegression
5. Country + SD                – XGBoost (randomised HP search, native cats)

Encoding
--------
- LR rows use pd.get_dummies one-hot encoding; missing values within a column
  become their own '__missing__' category so no observations are discarded.
- GBM rows use the native categorical encoding from `pipeline.gbm_encoding`:
  ordinal SD vars (income, education bracket, religiosity, …) are cast to
  numeric so the trees see their natural ordering, and nominal SD vars
  (gender, religion denomination, country, …) are passed as pd.Categorical
  to XGBoost (`enable_categorical=True`) instead of being dummy-coded.

The three optional covariates (religion, mnactic, isco08) are included when
the corresponding source column exists in the DB; if absent they're skipped.
"""

import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, ParameterSampler
from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .bootstrap import recode_demographics
from .config import MODEL_REGISTRY, COLOR_PALETTE
from .gbm_encoding import prepare_features

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter grid
# ─────────────────────────────────────────────────────────────────────────────

# XGBoost search space for the randomised hyperparameter search.  The ranges
# bracket the region where the selected parameters fall on this data, so the
# search is not pinned against a boundary; XGB_N_ITER sets how many
# combinations are sampled from the grid.
XGB_PARAM_DIST = {
    "learning_rate":     [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
    "max_depth":         [5, 6, 7, 8, 9],
    "min_child_weight":  [3, 5, 8, 13],
    "reg_alpha":         [0.001, 0.01, 0.1],
    "reg_lambda":        [0.1, 0.5, 1.0, 2.5],
    "subsample":         [0.7, 0.8, 0.9],
    "colsample_bytree":  [0.6, 0.7, 0.8],
    "gamma":             [0.0, 0.005, 0.01],
}
XGB_N_ITER = 18

CV_FOLDS    = 10
RANDOM_SEED = 42

# All SD covariates (optional ones skipped when source column is absent)
COVARIATE_VARS = [
    "gender_clean",
    "ethnicmajority_clean",
    "immigration_background",
    "education_bracket",
    "income_clean",
    "hincfel_clean",
    "fnsdfml_clean",
    "generations",
    "religiosity_clean",
    "domicil_clean",
    "political_interest_clean",
    # "political_involvement",
    "internet_time_bracket",
    "religion_clean",
    "mnactic_clean",
    "isco08_major_group",
]


# ─────────────────────────────────────────────────────────────────────────────
# Weighted metrics
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_r2(y, y_pred, w):
    w = np.asarray(w, float)
    y_mean = np.average(y, weights=w)
    ss_res = np.sum(w * (y - y_pred) ** 2)
    ss_tot = np.sum(w * (y - y_mean) ** 2)
    return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)


def _weighted_mse(y, y_pred, w):
    return float(np.average((y - y_pred) ** 2, weights=np.asarray(w, float)))


def _weighted_mae(y, y_pred, w):
    return float(np.average(np.abs(y - y_pred), weights=np.asarray(w, float)))


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost random search (inner 3-fold CV on training data, with early stopping)
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_random_search(X_df, y, w, n_iter=XGB_N_ITER, seed=RANDOM_SEED, n_jobs=-1):
    """Sample n_iter HP combinations, score each on a weighted 3-fold inner CV
    (with a 10% val slice carved out of each inner-train for early stopping),
    and return (best_params, median_best_iteration).  Caller uses those to
    refit a final XGBRegressor in the outer CV without further early stopping.
    """
    sampler = list(ParameterSampler(XGB_PARAM_DIST, n_iter=n_iter, random_state=seed))
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=seed)

    def _score_combo(params, combo_idx):
        # Per-(combo, fold) seeded RNG so the val-split permutation is
        # deterministic regardless of joblib thread interleaving.
        fold_scores, fold_iters = [], []
        for fold_i, (tr_idx, va_idx) in enumerate(inner_kf.split(X_df)):
            rng_local = np.random.RandomState(seed + 17 + combo_idx * 101 + fold_i)
            perm = rng_local.permutation(tr_idx)
            n_val = max(50, int(0.1 * len(perm)))
            val_i, tr2_i = perm[:n_val], perm[n_val:]
            m = XGBRegressor(
                **params,
                n_estimators=1000,
                tree_method="hist",
                enable_categorical=True,
                early_stopping_rounds=50,
                random_state=seed,
                n_jobs=1,                  # parallelism is at the outer Parallel level
                verbosity=0,
            )
            m.fit(
                X_df.iloc[tr2_i], y[tr2_i],
                sample_weight=w[tr2_i],
                eval_set=[(X_df.iloc[val_i], y[val_i])],
                sample_weight_eval_set=[w[val_i]],
                verbose=False,
            )
            fold_scores.append(
                _weighted_r2(y[va_idx], m.predict(X_df.iloc[va_idx]), w[va_idx])
            )
            fold_iters.append(int(m.best_iteration) + 1)
        return float(np.mean(fold_scores)), params, int(np.median(fold_iters))

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_score_combo)(p, i) for i, p in enumerate(sampler)
    )
    results.sort(key=lambda t: -t[0])
    _, best_params, best_n = results[0]
    return best_params, best_n


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_prepare(db_path, align_col: str, weight_col: str) -> pd.DataFrame:
    """Load alignment + ESS covariate data from DB, derive SD variables."""
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(ess11)")
    ess_cols = {r[1] for r in cursor.fetchall()}
    cursor.execute("PRAGMA table_info(demographics)")
    demo_cols = {r[1] for r in cursor.fetchall()}

    opt_ess  = [c for c in [
        "rlgdnm", "mnactic", "iscoco", "isco08", "isco08c", "occb",
        "rlgblg", "brncntr", "cntbrthd",
        "facntr", "fbrncntc", "mocntr", "mbrncntc",
        "hincfel", "fnsdfml",
    ] if c in ess_cols]
    opt_demo = [c for c in ["domicil_clean", "rlgdnm_clean"] if c in demo_cols]

    opt_ess_sel  = (", " + ", ".join(f"e.{c}" for c in opt_ess))  if opt_ess  else ""
    opt_demo_sel = (", " + ", ".join(f'd."{c}"' for c in opt_demo)) if opt_demo else ""

    query = f"""
        SELECT d."{align_col}" AS alignment, e.cntry, e.new_id,
               e.{weight_col},
               e.agea, e.edulvlb, e.nobingnd, e.feethngr, e.hinctnta,
               e.rlgdgr, e.netustm, e.domicil, e.polintr,
               e.contplt, e.donprty, e.badge, e.sgnptit, e.pbldmna,
               e.bctprd, e.pstplonl{opt_ess_sel}{opt_demo_sel}
        FROM demographics d
        JOIN ess11 e ON d.new_id = e.new_id
        WHERE d."{align_col}" IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df = recode_demographics(df)
    df = df.dropna(subset=["alignment", "cntry"])
    df["weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)
    return df


def _sd_dummies(df: pd.DataFrame) -> np.ndarray:
    """
    Dummy-code all available SD covariates via pd.get_dummies.
    Missing values become an explicit '__missing__' category.
    Variable name is used as the dummy prefix so identical category values
    across variables (e.g. '1.0' for both gender and income) do not collide.
    Returns dense float array.
    """
    cat_cols = [c for c in COVARIATE_VARS if c in df.columns]
    df_enc = df[cat_cols].astype(str).replace("nan", "__missing__")
    dummies = pd.get_dummies(df_enc)
    feature_names = sorted(dummies.columns.tolist())
    return dummies[feature_names].values.astype(float)


def _country_dummies(df: pd.DataFrame) -> np.ndarray:
    """Dummy-code country for the baseline."""
    return pd.get_dummies(df["cntry"].astype(str), prefix="cntry").values.astype(float)


def _country_sd_dummies(df: pd.DataFrame) -> np.ndarray:
    """Dummy-code country + all SD covariates."""
    ctry = pd.get_dummies(df["cntry"].astype(str), prefix="cntry")
    cat_cols = [c for c in COVARIATE_VARS if c in df.columns]
    df_enc = df[cat_cols].astype(str).replace("nan", "__missing__")
    sd = pd.get_dummies(df_enc)
    sd = sd[sorted(sd.columns.tolist())]
    return pd.concat([ctry, sd], axis=1).values.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _slice(X, idx):
    """Row-slice numpy array (LR rows) or pandas DataFrame (GBM rows)."""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _cv_evaluate(model_name, model_factory, X, y, w,
                 n_splits=CV_FOLDS, seed=RANDOM_SEED, n_jobs=-1) -> dict:
    """Run n_splits-fold weighted CV and return metric dict."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def _eval_fold(tr, va):
        m = model_factory()
        try:
            m.fit(_slice(X, tr), y[tr], sample_weight=w[tr])
        except TypeError:
            m.fit(_slice(X, tr), y[tr])
        y_pred_va = m.predict(_slice(X, va))
        y_pred_tr = m.predict(_slice(X, tr))
        return (
            _weighted_r2(y[va], y_pred_va, w[va]),
            _weighted_mse(y[va], y_pred_va, w[va]),
            _weighted_mae(y[va], y_pred_va, w[va]),
            _weighted_r2(y[tr], y_pred_tr, w[tr]),
        )

    fold_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_eval_fold)(tr, va) for tr, va in kf.split(X)
    )
    r2s, mses, maes, train_r2s = zip(*fold_results)

    return {
        "model":           model_name,
        "r2_mean":         float(np.mean(r2s)),
        "r2_std":          float(np.std(r2s)),
        "train_r2_mean":   float(np.mean(train_r2s)),
        "train_r2_std":    float(np.std(train_r2s)),
        "mse_mean":        float(np.mean(mses)),
        "mae_mean":        float(np.mean(maes)),
        "_fold_r2s":       list(r2s),
        "_fold_train_r2s": list(train_r2s),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-method runner (used for parallelism in run_method_comp)
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_method(
    method_name: str,
    X,                                # np.ndarray (LR rows, one-hot) or pd.DataFrame (GBM rows, native)
    y: np.ndarray,
    w: np.ndarray,
    use_gbm: bool,
    n_folds: int,
    seed: int,
    n_jobs_inner: int,
) -> dict:
    """
    Run optional XGBoost random search then CV for a single method.
    Designed to be called inside a Parallel block.
    """
    if use_gbm:
        best_p, best_n = _xgb_random_search(
            X, y, w, n_iter=XGB_N_ITER, seed=seed, n_jobs=n_jobs_inner,
        )
        def factory(p=best_p, n=best_n, s=seed):
            return XGBRegressor(
                **p, n_estimators=n,
                tree_method="hist", enable_categorical=True,
                random_state=s, n_jobs=1, verbosity=0,
            )
    else:
        best_p  = None
        factory = LinearRegression
    result = _cv_evaluate(
        method_name, factory, X, y, w, n_folds, seed, n_jobs=n_jobs_inner
    )
    return result, best_p


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_ORDER = [
    "Country-only (Linear)",
    "SD-only (Linear)",
    "SD-only (GBM)",
    "Country+SD (Linear)",
    "Country+SD (GBM)",
]
_MODEL_COLORS = COLOR_PALETTE[:len(_MODEL_ORDER)]


def _plot_r2_all_models(
    all_results: list[dict],
    output_dir: Path,
    weight_col: str,
    question_set_key: str = "",
) -> None:
    """
    Grouped bar chart: x-axis = methods, grouped bars = LLM models.
    Each method shows one bar per LLM model, side-by-side.
    """
    if not all_results:
        return

    df = pd.DataFrame(all_results)   # columns: method, r2_mean, r2_std, llm_model

    llm_models = df["llm_model"].unique().tolist()
    n_models   = len(llm_models)
    n_methods  = len(_MODEL_ORDER)

    # Colour palette – one colour per LLM model
    model_colors = {m: COLOR_PALETTE[i % len(COLOR_PALETTE)]
                    for i, m in enumerate(llm_models)}

    total_width = 0.8
    bar_w = total_width / n_models
    x = np.arange(n_methods)

    fig, ax = plt.subplots(figsize=(max(12, 2.5 * n_methods), 7))

    for i, llm in enumerate(llm_models):
        sub = df[df["llm_model"] == llm].set_index("method").reindex(_MODEL_ORDER)
        offsets = x - total_width / 2 + bar_w * (i + 0.5)
        bars = ax.bar(
            offsets,
            sub["r2_mean"],
            width=bar_w * 0.9,
            yerr=sub["r2_std"],
            capsize=5,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
            color=model_colors[llm],
            label=llm,
        )
        for rect, (mean_val, std_val) in zip(
                bars, zip(sub["r2_mean"], sub["r2_std"])):
            if not np.isnan(mean_val):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    mean_val + (std_val if not np.isnan(std_val) else 0) + 0.003,
                    f"{mean_val:.3f}",
                    ha="center", va="bottom",
                    fontsize=7, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(_MODEL_ORDER, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel(
        f"Weighted Test R\u00b2 ({weight_col}, 10-fold CV)",
        fontsize=12, fontweight="bold",
    )
    qs_label = f"  [{question_set_key}]" if question_set_key else ""
    ax.set_title(
        f"Test R\u00b2 by Method \u2014 All Models ({weight_col}){qs_label}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(title="LLM model", fontsize=10, title_fontsize=11)

    plt.tight_layout()
    out_path = output_dir / f"method_comp_all_models_r2.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined R² plot saved: {out_path}")


def _plot_r2_by_llm(
    all_results: list[dict],
    output_dir: Path,
    weight_col: str,
    question_set_key: str = "",
) -> None:
    """
    Grouped bar chart: x-axis = LLM models, grouped bars = statistical methods.
    The complement of _plot_r2_all_models.
    Saved as method_comp_all_models_r2_by_llm.png.
    """
    if not all_results:
        return

    df = pd.DataFrame(all_results)   # columns: method, r2_mean, r2_std, llm_model

    registry_models = set(MODEL_REGISTRY.values())
    llm_models = sorted([m for m in df["llm_model"].unique().tolist() if m in registry_models])
    n_llm      = len(llm_models)
    n_methods  = len(_MODEL_ORDER)

    method_colors = dict(zip(_MODEL_ORDER, _MODEL_COLORS))

    total_width = 0.8
    bar_w = total_width / n_methods
    x = np.arange(n_llm)

    fig, ax = plt.subplots(figsize=(max(10, 1.8 * n_llm), 5))

    for i, method in enumerate(_MODEL_ORDER):
        sub = (
            df[df["method"] == method]
            .set_index("llm_model")
            .reindex(llm_models)
        )
        offsets = x - total_width / 2 + bar_w * (i + 0.5)
        bars = ax.bar(
            offsets,
            sub["r2_mean"],
            width=bar_w * 0.9,
            yerr=sub["r2_std"],
            capsize=5,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
            color=method_colors[method],
            label=method,
        )
        for rect, (mean_val, std_val) in zip(
                bars, zip(sub["r2_mean"], sub["r2_std"])):
            if not (np.isnan(mean_val) if isinstance(mean_val, float) else False):
                try:
                    mv = float(mean_val)
                    sv = float(std_val) if not np.isnan(float(std_val)) else 0.0
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
    ax.set_ylabel(
        f"Weighted Test R\u00b2 ({weight_col}, 10-fold CV)",
        fontsize=14, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=13)
    qs_label = f"  [{question_set_key}]" if question_set_key else ""
    ax.set_title(
        f"Test R\u00b2 by LLM Model \u2014 All Methods ({weight_col}){qs_label}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=13, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    out_path = output_dir / "method_comp_all_models_r2_by_llm.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined R\u00b2 by-LLM plot saved: {out_path}")


def _plot_method_comp(
    df_res: pd.DataFrame,
    df_folds: pd.DataFrame,
    output_dir: Path,
    model_short: str,
    weight_col: str,
    question_set_key: str = "",
) -> None:
    """Two-panel figure: test-R² bar chart + train-vs-test comparison."""
    summary = df_res.set_index("model").reindex(_MODEL_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    qs_label = f"  [{question_set_key}]" if question_set_key else ""

    # ── Panel 1: Test R² bar chart with error bars ───────────────────────────
    ax = axes[0]
    x = np.arange(len(_MODEL_ORDER))
    ax.bar(
        x, summary["r2_mean"], yerr=summary["r2_std"],
        capsize=8, alpha=0.85, edgecolor="black", linewidth=1.2,
        color=_MODEL_COLORS,
    )
    for i, (mean_val, std_val) in enumerate(
            zip(summary["r2_mean"], summary["r2_std"])):
        if not np.isnan(mean_val):
            ax.text(
                i, mean_val + std_val + 0.005, f"{mean_val:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(_MODEL_ORDER, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(
        f"Weighted Test R\u00b2 ({weight_col}, 10-fold CV)",
        fontsize=12, fontweight="bold",
    )
    ax.set_title(
        f"{model_short}{qs_label} \u2014 Test R\u00b2 by Method",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_axisbelow(True)

    # ── Panel 2: Train vs Test R² ─────────────────────────────────────────────
    ax = axes[1]
    width = 0.35
    train_means = summary["train_r2_mean"].values
    test_means  = summary["r2_mean"].values
    ax.bar(
        x - width / 2, train_means, width,
        label="Train", alpha=0.75, edgecolor="black", color=_MODEL_COLORS,
    )
    ax.bar(
        x + width / 2, test_means, width,
        label="Test", alpha=0.45, edgecolor="black", color=_MODEL_COLORS,
        hatch="//",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(_MODEL_ORDER, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(
        f"Weighted R\u00b2 ({weight_col})",
        fontsize=12, fontweight="bold",
    )
    ax.set_title(
        f"{model_short}{qs_label} \u2014 Train vs Test R\u00b2",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=11)

    plt.tight_layout()
    out_path = output_dir / f"method_comp_{model_short}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_method_comp(
    db_path,
    question_set_key: str,
    model_registry: dict = MODEL_REGISTRY,
    output_dir: Path = None,
    weight_col: str = "pspwght",
    n_folds: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Compare 5 prediction approaches for each LLM alignment score.

    Results written to output_dir:
      method_comp_{model_short}.csv   – CV metrics per method
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(ess11)")
    ess_cols = {r[1] for r in cursor.fetchall()}
    cursor.execute("PRAGMA table_info(demographics)")
    demo_cols = {r[1] for r in cursor.fetchall()}
    conn.close()

    if weight_col not in ess_cols:
        warnings.warn(
            f"Weight column '{weight_col}' not in ess11; falling back to equal weights."
        )
        weight_col = None

    all_model_results: list[dict] = []
    covariate_log: list[dict] = []   # SD variable audit per model

    for model_full, model_short in model_registry.items():
        align_col = f"{model_short}_{question_set_key}_alignment"
        if align_col not in demo_cols:
            print(f"  Skipping method comp for {model_short} (no alignment column)")
            continue

        print(f"\n  Method comparison: {model_short}")
        print(f"  Alignment column:  {align_col}")
        print(f"  Weight column:     {weight_col or 'uniform'}")

        wc = weight_col if weight_col else "pspwght"
        try:
            df = _load_and_prepare(db_path, align_col, wc)
        except Exception as e:
            warnings.warn(f"  Failed to load data for {model_short}: {e}")
            continue

        if weight_col is None:
            df["weight"] = 1.0

        y = df["alignment"].values
        w = df["weight"].values

        X_sd      = _sd_dummies(df)
        X_ctry    = _country_dummies(df)
        X_sd_ctry = _country_sd_dummies(df)
        # Native-encoded DataFrames for the two GBM rows: ordinal SD vars cast
        # to numeric, nominal vars cast to pd.Categorical so XGBoost handles
        # them via enable_categorical=True instead of one-hot dummies.
        X_sd_native      = prepare_features(df, COVARIATE_VARS, with_country=False, encoding="native").X
        X_sd_ctry_native = prepare_features(df, COVARIATE_VARS, with_country=True,  encoding="native").X

        sd_vars_used = [c for c in COVARIATE_VARS if c in df.columns]
        sd_vars_missing = [c for c in COVARIATE_VARS if c not in df.columns]
        countries_used = sorted(df["cntry"].dropna().unique().tolist())
        print(f"    N = {len(y):,}  |  SD features = {X_sd.shape[1]}  |  Country+SD = {X_sd_ctry.shape[1]}")
        print(f"    SD variables used: {sd_vars_used}")

        covariate_log.append({
            "llm_model":        model_short,
            "n_respondents":    int(len(y)),
            "sd_vars_used":     sd_vars_used,
            "sd_vars_missing":  sd_vars_missing,
            "n_sd_dummies":     int(X_sd.shape[1]),
            "n_country_sd_dummies": int(X_sd_ctry.shape[1]),
            "n_countries":      len(countries_used),
            "countries":        countries_used,
        })

        # ── Run all 5 methods in parallel ──────────────────────────────────────
        # Each method is independent; n_jobs_inner controls per-method parallelism.
        # With 5 outer workers, limit inner to avoid over-subscribing cores.
        import os
        n_cores = os.cpu_count() or 4
        n_outer = 5
        n_jobs_inner = max(1, n_cores // n_outer)

        _method_tasks = [
            ("Country-only (Linear)", X_ctry,            False),
            ("SD-only (Linear)",      X_sd,              False),
            ("SD-only (GBM)",         X_sd_native,       True),
            ("Country+SD (Linear)",   X_sd_ctry,         False),
            ("Country+SD (GBM)",      X_sd_ctry_native,  True),
        ]

        print(f"    Running {len(_method_tasks)} methods in parallel "
              f"(n_outer={n_outer}, n_jobs_inner={n_jobs_inner})...")

        _raw_results = Parallel(n_jobs=n_outer, prefer="threads")(
            delayed(_run_one_method)(
                name, X, y, w, use_gbm, n_folds, seed, n_jobs_inner
            )
            for name, X, use_gbm in _method_tasks
        )

        results = []
        for (res, best_p), (name, _, use_gbm) in zip(_raw_results, _method_tasks):
            if use_gbm:
                print(f"      {name} best params: {best_p}")
            results.append(res)

        # Build fold-level DataFrame for plotting
        fold_rows = []
        for res in results:
            for fold_idx, (tr2, te2) in enumerate(
                    zip(res["_fold_train_r2s"], res["_fold_r2s"]), 1):
                fold_rows.append({
                    "model": res["model"],
                    "fold": fold_idx,
                    "test_r2": te2,
                    "train_r2": tr2,
                })
        df_folds = pd.DataFrame(fold_rows)

        # Strip private keys before saving summary CSV
        df_res = pd.DataFrame([
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ])
        df_res["llm_model"]    = model_short
        df_res["question_set"] = question_set_key
        df_res["weight_col"]   = weight_col or "uniform"
        df_res["timestamp"]    = datetime.now().isoformat()

        out_path = output_dir / f"method_comp_{model_short}.csv"
        df_res.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        print(df_res[["model", "r2_mean", "r2_std", "mse_mean"]].to_string(index=False))

        _plot_method_comp(df_res, df_folds, output_dir, model_short, weight_col or "uniform", question_set_key)

        # Collect rows for the combined all-models plot
        for res in results:
            all_model_results.append({
                "method":    res["model"],
                "r2_mean":   res["r2_mean"],
                "r2_std":    res["r2_std"],
                "llm_model": model_short,
            })

    _plot_r2_all_models(all_model_results, output_dir, weight_col or "uniform", question_set_key)
    _plot_r2_by_llm(all_model_results, output_dir, weight_col or "uniform", question_set_key)

    # ── Covariate audit log ────────────────────────────────────────────────────
    if covariate_log:
        import json
        log = {
            "question_set_key":  question_set_key,
            "weight_col":        weight_col or "uniform",
            "cv_folds":          n_folds,
            "random_seed":       seed,
            "timestamp":         datetime.now().isoformat(),
            "covariate_vars_requested": COVARIATE_VARS,
            "models": covariate_log,
        }
        log_path = output_dir / "covariates_used.json"
        with open(log_path, "w") as fh:
            json.dump(log, fh, indent=2)
        print(f"  Covariate log saved: {log_path}")


def _plot_train_test_grid(
    df_all: pd.DataFrame,
    output_dir: Path,
    weight_col: str,
    question_set_key: str = "",
) -> None:
    """
    Grid of train-vs-test R² plots – one subplot per LLM model, 2 columns.
    Styling mirrors panel 2 of _plot_method_comp.
    """
    registry_models = set(MODEL_REGISTRY.values())
    llm_models = sorted([m for m in df_all["llm_model"].unique() if m in registry_models])
    if not llm_models:
        llm_models = sorted(df_all["llm_model"].unique().tolist())

    n_models = len(llm_models)
    n_cols = 2
    n_rows = (n_models + 1) // n_cols

    method_colors = dict(zip(_MODEL_ORDER, _MODEL_COLORS))
    x = np.arange(len(_MODEL_ORDER))
    width = 0.35
    qs_label = f"  [{question_set_key}]" if question_set_key else ""

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(9 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for idx, llm in enumerate(llm_models):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        sub = (
            df_all[df_all["llm_model"] == llm]
            .set_index("model")
            .reindex(_MODEL_ORDER)
        )

        train_means = sub["train_r2_mean"].values.astype(float)
        test_means  = sub["r2_mean"].values.astype(float)
        train_stds  = sub["train_r2_std"].values.astype(float)
        test_stds   = sub["r2_std"].values.astype(float)

        colors = [method_colors[m] for m in _MODEL_ORDER]

        ax.bar(
            x - width / 2, train_means, width,
            yerr=train_stds, capsize=5,
            label="Train", alpha=0.75, edgecolor="black", linewidth=0.8,
            color=colors,
        )
        ax.bar(
            x + width / 2, test_means, width,
            yerr=test_stds, capsize=5,
            label="Test", alpha=0.45, edgecolor="black", linewidth=0.8,
            color=colors, hatch="//",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(_MODEL_ORDER, rotation=45, ha="right", fontsize=13)
        ax.set_ylabel(
            f"Weighted R\u00b2 ({weight_col})",
            fontsize=14, fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=13)
        ax.set_title(llm, fontsize=15, fontweight="bold", pad=10)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(fontsize=13)

    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    out_path = output_dir / "method_comp_train_test_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Train/test grid plot saved: {out_path}")


def regen_method_comp_plots(mc_dir: Path) -> None:
    """
    Regenerate _plot_r2_all_models, _plot_r2_by_llm, and the train/test
    grid from existing method_comp_*.csv files in *mc_dir*, without
    rerunning CV.
    """
    csv_files = sorted(mc_dir.glob("method_comp_*.csv"))
    if not csv_files:
        return

    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            pass
    if not frames:
        return

    df_all = pd.concat(frames, ignore_index=True)
    # Infer metadata from first row that has it
    weight_col       = str(df_all["weight_col"].iloc[0])   if "weight_col"   in df_all.columns else "uniform"
    question_set_key = str(df_all["question_set"].iloc[0]) if "question_set" in df_all.columns else ""

    all_results = [
        {"method": row["model"], "r2_mean": row["r2_mean"],
         "r2_std": row["r2_std"], "llm_model": row["llm_model"]}
        for _, row in df_all.iterrows()
        if not pd.isna(row.get("r2_mean"))
    ]

    print(f"    Regenerating method-comp combined plots for {question_set_key!r} …")
    _plot_r2_all_models(all_results, mc_dir, weight_col, question_set_key)
    _plot_r2_by_llm(all_results, mc_dir, weight_col, question_set_key)
    _plot_train_test_grid(df_all, mc_dir, weight_col, question_set_key)

