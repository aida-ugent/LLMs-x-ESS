"""
Inverse Propensity Weighting (IPW) analysis of country vs sociodemographic
effects on LLM alignment scores.

Entry point
-----------
run_ipw(db_path, question_set_key, ...) -> writes all artefacts to
    results/{qset}/ipw/

Pipeline
------------------------------------
1. Load + recode (multi-model)
2. Compute cross-model mean deviation outcome
3. Cross-fitted multinomial GBM propensity
4. Three absolute-propensity settings on Phat(C=c|X):
       clip_p01 (main)        : clip propensity from below at 0.01
       drop_p01 (robustness)  : drop respondents with Phat < 0.01
       drop_p05 (robustness)  : drop respondents with Phat < 0.05
5. Diagnostics (propensity quality, weights, SD balance)
6. Point estimates: per-LLM between-country variance + country means
7. Stratified bootstrap with cross-fit refit (B=100); each replicate
   emits per-(LLM, setting) σ_pre / σ_post so the variance tables get
   95% CIs on Δσ = σ_post − σ_pre.
8. Paper-ready artefacts: country-mean three-stage plots, clip-only
   cross-model-deviation before-after plot, two LaTeX variance tables
   (clip + drop robustness).

Estimand
--------
Country means under the pooled-European SD distribution.
Weight per respondent i in country c_i:
    w_ipw_i    = P(C=c_i) / Phat(C=c_i | X_i)
    w_final_i  = pspwght_i * w_ipw_i
then per-country normalised so the country's weights sum to the design-
weighted country sample size.

Primary outcome
---------------
Cross-model mean deviation:
    dev_i = mean_m ( alignment_i_m  -  E[alignment_m]  )
where E[alignment_m] is the FULL-SAMPLE pspwght-weighted population mean
of alignment for model m, held fixed (not recomputed under reweighting).
"""
from __future__ import annotations

import json
import sqlite3
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from .bootstrap import recode_demographics
from .config import MODEL_REGISTRY
from .gbm_encoding import prepare_features
from .method_comp import COVARIATE_VARS

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# SD covariates used in the IPW propensity model.
IPW_SD_COVARIATES: list[str] = list(COVARIATE_VARS)

# Cross-fitting / bootstrap defaults
CROSSFIT_FOLDS = 5
BOOTSTRAP_REPLICATES_DEFAULT = 100   # matches --ipw-n-bootstrap CLI default
RANDOM_SEED = 42

# XGBoost hyperparameters for the multinomial propensity model.
XGB_PROPENSITY_PARAMS = {
    "learning_rate":    0.05,
    "max_depth":        7,
    "min_child_weight": 5,
    "reg_alpha":        0.01,
    "reg_lambda":       1.0,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "gamma":            0.005,
    "n_estimators":     400,
}

# Absolute-propensity thresholds for the three analyses.
# Operates on the own-country propensity Phat(C=c|X), not on the weight.
# A low propensity drives a large IPW weight; thresholding the propensity
# directly is interpretable ("respondents with <1% propensity for their
# country given X") and applies the same rule across countries.
SETTING_DEFS: list[tuple[str, str, float]] = [
    ("clip_p01", "clip", 0.01),   # main: floor propensity at 0.01
    ("drop_p01", "drop", 0.01),   # robustness 1: drop prop < 0.01
    ("drop_p05", "drop", 0.05),   # robustness 2: drop prop < 0.05
]
SETTING_NAMES   = [s[0] for s in SETTING_DEFS]
PRIMARY_SETTING = "clip_p01"

# SMD interpretation thresholds (Stuart 2010, adapted)
SMD_GOOD       = 0.10
SMD_ACCEPTABLE = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ipw_dataset(
    db_path: Path,
    question_set_key: str,
    model_registry: dict = MODEL_REGISTRY,
    weight_col: str = "pspwght",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load a single per-respondent DataFrame with: new_id, cntry, pspwght,
    every SD raw column needed by recode_demographics, and one alignment
    column per LLM (named `align_{model_short}`).

    Returns (df, model_shorts_present).  Only models with an existing
    alignment column for this question set are included.

    Rows with missing alignment for ALL models are dropped.
    Rows with missing country are dropped.
    pspwght missing -> 1.0.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(ess11)")
    ess_cols = {r[1] for r in cur.fetchall()}
    cur.execute("PRAGMA table_info(demographics)")
    demo_cols = {r[1] for r in cur.fetchall()}

    # Identify which model alignment columns are present
    model_shorts_present: list[str] = []
    for model_full, model_short in model_registry.items():
        col = f"{model_short}_{question_set_key}_alignment"
        if col in demo_cols:
            model_shorts_present.append(model_short)
    if not model_shorts_present:
        conn.close()
        raise RuntimeError(
            f"No alignment columns for question set {question_set_key!r} in demographics."
        )

    align_select = ", ".join(
        f'd."{ms}_{question_set_key}_alignment" AS align_{ms}'
        for ms in model_shorts_present
    )

    # Raw ESS columns needed by recode_demographics + a few optionals
    ess_required = [
        "agea", "edulvlb", "nobingnd", "feethngr", "hinctnta",
        "rlgdgr", "netustm", "domicil", "polintr",
        "contplt", "donprty", "badge", "sgnptit", "pbldmna",
        "bctprd", "pstplonl",
    ]
    ess_optional = [
        "rlgdnm", "rlgblg", "mnactic", "iscoco", "isco08", "isco08c", "occb",
        "brncntr", "cntbrthd", "facntr", "fbrncntc", "mocntr", "mbrncntc",
        "hincfel", "fnsdfml",
    ]
    ess_present = [c for c in ess_required + ess_optional if c in ess_cols]
    ess_select  = ", ".join(f"e.{c}" for c in ess_present)

    demo_optional = [c for c in ["domicil_clean", "rlgdnm_clean"] if c in demo_cols]
    demo_extra    = (", " + ", ".join(f'd."{c}"' for c in demo_optional)) if demo_optional else ""

    weight_select = f"e.{weight_col}" if weight_col in ess_cols else "1.0 AS pspwght"

    query = f"""
        SELECT d.new_id,
               e.cntry,
               {weight_select},
               {ess_select},
               {align_select}
               {demo_extra}
        FROM demographics d
        JOIN ess11 e ON d.new_id = e.new_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df = recode_demographics(df)

    # Drop rows missing country
    df = df.dropna(subset=["cntry"]).copy()

    # Drop rows where ALL alignment columns are missing
    align_cols = [f"align_{ms}" for ms in model_shorts_present]
    df = df[df[align_cols].notna().any(axis=1)].copy()

    # Sanitise weight
    if weight_col in df.columns:
        df["pspwght"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)
    if "pspwght" not in df.columns:
        df["pspwght"] = 1.0
    df["pspwght"] = df["pspwght"].astype(float).clip(lower=0.0)

    df = df.reset_index(drop=True)
    return df, model_shorts_present


# ─────────────────────────────────────────────────────────────────────────────
# Outcome: cross-model mean deviation
# ─────────────────────────────────────────────────────────────────────────────

def compute_cross_model_deviation(
    df: pd.DataFrame,
    model_shorts: list[str],
) -> tuple[pd.Series, dict[str, float]]:
    """
    For each respondent, compute the mean across models m of
        alignment_i_m  -  Ehat[alignment_m]
    where Ehat[alignment_m] is the FULL-SAMPLE pspwght-weighted population
    mean of alignment for model m.  It is held fixed rather than recomputed
    under reweighting, so the outcome scale does not shift between settings.

    Missing per-(person, model) entries are skipped in the per-respondent
    average.  Respondents with all-missing alignment are NaN.

    Returns:
        deviation: pd.Series indexed like df, name='cross_model_dev'
        ref_means: dict[model_short -> Ehat[alignment_m]]
    """
    w = df["pspwght"].values.astype(float)
    ref_means: dict[str, float] = {}
    devs = np.full((len(df), len(model_shorts)), np.nan, dtype=float)
    for k, ms in enumerate(model_shorts):
        a = df[f"align_{ms}"].values.astype(float)
        mask = np.isfinite(a)
        if not mask.any():
            ref_means[ms] = np.nan
            continue
        mu = float(np.average(a[mask], weights=w[mask]))
        ref_means[ms] = mu
        devs[mask, k] = a[mask] - mu

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_dev = np.nanmean(devs, axis=1)
    return pd.Series(mean_dev, index=df.index, name="cross_model_dev"), ref_means


# ─────────────────────────────────────────────────────────────────────────────
# Cross-fitted multinomial propensity
# ─────────────────────────────────────────────────────────────────────────────

def _country_codes(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (country_idx: N int array, classes: sorted unique country list)."""
    classes = sorted(df["cntry"].dropna().astype(str).unique().tolist())
    code = {c: i for i, c in enumerate(classes)}
    idx = df["cntry"].astype(str).map(code).values.astype(int)
    return idx, classes


def fit_crossfitted_propensity(
    df: pd.DataFrame,
    n_splits: int = CROSSFIT_FOLDS,
    params: dict | None = None,
    seed: int = RANDOM_SEED,
    n_jobs: int = 1,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Cross-fitted multinomial XGBoost over countries.

    Fit is sample-weighted by ``pspwght`` so the estimated propensity
    targets the design-weighted population P(C=c|X), matching the paper's
    "(weighted) estimate of the probability of being located in country c_p
    given characteristics X_p".  Per-fold log-loss is reported under the
    same weighting for consistency with the optimized loss.  Hyperparameters
    come from ``XGB_PROPENSITY_PARAMS`` (overridable via ``params``).

    Returns:
        oof_proba: (N, K) array of out-of-fold uncalibrated probabilities,
                   rows summing to 1.
        classes  : list of country codes corresponding to the K columns.
        meta     : dict with per-fold log-loss + estimator info.
    """
    y, classes = _country_codes(df)
    sw = df["pspwght"].values.astype(float)
    K = len(classes)
    N = len(df)
    oof = np.zeros((N, K), dtype=float)

    params = {**XGB_PROPENSITY_PARAMS, **(params or {})}
    enc = prepare_features(df, IPW_SD_COVARIATES, with_country=False, encoding="native")
    X = enc.X  # pd.DataFrame with categoricals

    # Group by new_id so duplicate respondents (produced by bootstrap with
    # replacement) end up in the SAME fold.  Otherwise a respondent's copies
    # would be split across train/test, leaking exact (X, C) into the OOF
    # predictions and shrinking bootstrap CIs.  On the full-sample run every
    # new_id is unique, so this reduces to ordinary StratifiedKFold.
    groups = df["new_id"].values
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_meta: list[dict] = []

    for fold_i, (tr, te) in enumerate(skf.split(np.arange(N), y, groups=groups)):
        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=K,
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
            **params,
        )
        clf.fit(X.iloc[tr], y[tr], sample_weight=sw[tr])
        proba = clf.predict_proba(X.iloc[te])

        clf_classes = np.asarray(clf.classes_)
        if not np.array_equal(clf_classes, np.arange(K)):
            order = np.argsort(clf_classes)
            proba = proba[:, order]
        oof[te] = proba

        try:
            ll = float(log_loss(y[te], proba, labels=list(range(K)),
                                sample_weight=sw[te]))
        except Exception:
            ll = float("nan")
        fold_meta.append({"fold": fold_i, "log_loss": ll, "n_test": int(len(te))})

    meta = {
        "params":      params,
        "n_splits":    n_splits,
        "fold_log_loss": fold_meta,
        "mean_log_loss": float(np.nanmean([m["log_loss"] for m in fold_meta])),
        "n_classes":   K,
        "n_obs":       N,
        "propensity_fit_weighted":  True,
        "weight_col":               "pspwght",
        "propensity_estimator":     "xgb",
    }
    return oof, classes, meta


# ─────────────────────────────────────────────────────────────────────────────
# Weight construction
# ─────────────────────────────────────────────────────────────────────────────

def compute_raw_ipw_weights(
    country_idx: np.ndarray,
    prop_cal: np.ndarray,
    pspwght: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute raw IPW weight and final_weight_raw = pspwght * ipw_weight.

    Country marginals P(C=c) are design-weighted (using pspwght).

    Returns (ipw_weight, final_weight_raw, country_marginal).
    """
    N, K = prop_cal.shape
    pspwght = np.asarray(pspwght, dtype=float)
    marginals = np.zeros(K, dtype=float)
    total_w = float(pspwght.sum())
    if total_w <= 0:
        total_w = float(N)
        pspwght = np.ones(N, dtype=float)
    for k in range(K):
        marginals[k] = float(pspwght[country_idx == k].sum()) / total_w

    own_prop     = prop_cal[np.arange(N), country_idx]
    own_marginal = marginals[country_idx]
    ipw_weight   = own_marginal / np.clip(own_prop, eps, 1.0)
    final_weight_raw = pspwght * ipw_weight
    return ipw_weight, final_weight_raw, marginals


def _renormalise_per_country(
    w: np.ndarray,
    country_idx: np.ndarray,
    K: int,
    target_per_country: np.ndarray,
) -> np.ndarray:
    """Rescale w within each country so it sums to target_per_country[c]."""
    out = w.copy()
    for k in range(K):
        mask = country_idx == k
        s = out[mask].sum()
        if s > 0 and target_per_country[k] > 0:
            out[mask] = out[mask] * (target_per_country[k] / s)
    return out


def apply_propensity_clip(
    prop_own: np.ndarray,
    own_marginal: np.ndarray,
    pspwght: np.ndarray,
    country_idx: np.ndarray,
    K: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Floor the own-country propensity at ``threshold`` (absolute value),
    recompute IPW weights with the clipped propensity, then renormalise the
    final weights per country to the design-weighted country sample size.

    Returns:
        final_weight : (N,) array of stabilised final weights
        affected     : (N,) bool — True if ``prop_own < threshold`` (propensity
                       was floored; the resulting weight is at the per-country cap).
    """
    affected = prop_own < threshold
    prop_clipped = np.maximum(prop_own, threshold)
    ipw_weight = own_marginal / prop_clipped
    final_weight_raw = pspwght * ipw_weight
    target_per_country = np.zeros(K, dtype=float)
    for k in range(K):
        target_per_country[k] = float(pspwght[country_idx == k].sum())
    final_weight = _renormalise_per_country(
        final_weight_raw, country_idx, K, target_per_country,
    )
    return final_weight, affected


def apply_propensity_drop(
    prop_own: np.ndarray,
    own_marginal: np.ndarray,
    pspwght: np.ndarray,
    country_idx: np.ndarray,
    K: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop respondents whose own-country propensity is below ``threshold``
    (absolute value); renormalise the survivors per country so the kept
    weights sum to the original design-weighted country target.

    Returns:
        final_weight : (N,) array; 0 for dropped respondents.
        affected     : (N,) bool — True if ``prop_own < threshold`` (dropped).
    """
    affected = prop_own < threshold
    safe_prop = np.where(affected, 1.0, prop_own)  # avoid div-by-tiny; row zeroed below
    ipw_weight = own_marginal / np.clip(safe_prop, 1e-12, 1.0)
    final_weight_raw = pspwght * ipw_weight
    final_weight_raw[affected] = 0.0
    target_per_country = np.zeros(K, dtype=float)
    for k in range(K):
        target_per_country[k] = float(pspwght[country_idx == k].sum())
    final_weight = _renormalise_per_country(
        final_weight_raw, country_idx, K, target_per_country,
    )
    return final_weight, affected


def _compute_setting_weights(
    prop_own: np.ndarray,
    own_marginal: np.ndarray,
    pspwght: np.ndarray,
    country_idx: np.ndarray,
    K: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Compute final-weight arrays + affected masks for every setting in
    ``SETTING_DEFS``.

    Returns:
        weights  : dict[setting_name -> (N,) final_weight]
        affected : dict[setting_name -> (N,) bool, True if below threshold]
    """
    weights:  dict[str, np.ndarray] = {}
    affected: dict[str, np.ndarray] = {}
    for setting, op, thr in SETTING_DEFS:
        if op == "clip":
            fw, aff = apply_propensity_clip(
                prop_own, own_marginal, pspwght, country_idx, K, thr,
            )
        elif op == "drop":
            fw, aff = apply_propensity_drop(
                prop_own, own_marginal, pspwght, country_idx, K, thr,
            )
        else:
            raise ValueError(f"unknown op {op!r} in SETTING_DEFS")
        weights[setting]  = fw
        affected[setting] = aff
    return weights, affected


def kish_ess_per_country(
    w: np.ndarray,
    country_idx: np.ndarray,
    K: int,
) -> np.ndarray:
    """Per-country Kish effective sample size."""
    ess = np.zeros(K, dtype=float)
    for k in range(K):
        wk = w[country_idx == k]
        wk = wk[wk > 0]
        if len(wk) == 0:
            ess[k] = 0.0
            continue
        s1 = float(wk.sum())
        s2 = float((wk ** 2).sum())
        ess[k] = (s1 * s1) / s2 if s2 > 0 else 0.0
    return ess


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics: propensity quality
# ─────────────────────────────────────────────────────────────────────────────

def propensity_metrics_overall(
    proba: np.ndarray,
    country_idx: np.ndarray,
    classes: list[str],
    pspwght: np.ndarray,
) -> dict:
    """
    Headline multinomial classifier metrics on out-of-fold proba.

    All metrics are **design-weighted by ``pspwght``** for consistency with
    the propensity fit (which is sample-weighted) and the IPW estimand.
    Per-fold log-loss lives in ``propensity_model_meta.json`` — not duplicated
    here.
    """
    K = len(classes)
    N = len(country_idx)
    y = country_idx
    y_pred = np.argmax(proba, axis=1)
    sw = np.asarray(pspwght, float)

    top1 = float(np.average((y_pred == y).astype(float), weights=sw))
    if K >= 3:
        top3_pred = np.argpartition(-proba, kth=2, axis=1)[:, :3]
        top3_hits = np.any(top3_pred == y[:, None], axis=1).astype(float)
        top3 = float(np.average(top3_hits, weights=sw))
    else:
        top3 = top1

    try:
        roc_macro = float(roc_auc_score(
            y, proba, multi_class="ovr", average="macro",
            labels=list(range(K)), sample_weight=sw,
        ))
    except Exception:
        roc_macro = float("nan")

    return {
        "top1_accuracy":     top1,
        "top3_accuracy":     top3,
        "roc_auc_ovr_macro": roc_macro,
        "n_obs":             int(N),
        "n_classes":         int(K),
        "weighted":          True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics: balance (SMD)
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & (w > 0)
    if not m.any():
        return float("nan")
    return float(np.average(x[m], weights=w[m]))


def _weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & (w > 0)
    if m.sum() < 2:
        return float("nan")
    mu = np.average(x[m], weights=w[m])
    var = np.average((x[m] - mu) ** 2, weights=w[m])
    return float(np.sqrt(max(var, 0.0)))


def _is_continuous_var(s: pd.Series) -> bool:
    """Treat as continuous/ordinal if dtype is numeric."""
    if pd.api.types.is_numeric_dtype(s):
        return True
    return False


def _categorical_dist(sv_subset: np.ndarray, w_subset: np.ndarray) -> dict[str, float]:
    """Weighted category shares for a single subset of rows."""
    w = np.asarray(w_subset, float)
    tot = float(w.sum())
    if tot <= 0:
        return {}
    cats = sorted(np.unique(sv_subset).tolist())
    return {cat: float(w[sv_subset == cat].sum() / tot) for cat in cats}


def _categorical_smd_from_dists(d_target: dict, d_other: dict) -> float:
    """Mean over the union of categories of |share_other - share_target|."""
    cats = set(d_target) | set(d_other)
    if not cats:
        return float("nan")
    return float(np.mean([
        abs(d_other.get(c, 0.0) - d_target.get(c, 0.0)) for c in cats
    ]))


def compute_balance_table(
    df: pd.DataFrame,
    country_idx: np.ndarray,
    classes: list[str],
    w_pre: np.ndarray,
    w_post: np.ndarray,
) -> pd.DataFrame:
    """
    For each (country, SD variable) compute absolute SMD against the pooled
    European weighted target distribution, both pre- and post-IPW.

    Continuous vars: |mu_c - mu_pool| / sigma_pool
    Categorical vars: mean over categories of |p_c - p_pool|

    Target reference is the pre-IPW (pspwght-weighted) full-sample distribution
    and is held fixed across the pre/post comparison, so any change in SMD is
    attributable to the reweighting rather than to a moving target.
    """
    rows = []
    w_pre  = np.asarray(w_pre, float)
    w_post = np.asarray(w_post, float)

    for var in IPW_SD_COVARIATES:
        if var not in df.columns:
            continue
        s = df[var]
        cont = _is_continuous_var(s)

        if cont:
            x = pd.to_numeric(s, errors="coerce").values.astype(float)
            mu_pool = _weighted_mean(x, w_pre)
            sd_pool = _weighted_std(x, w_pre)
            if not np.isfinite(sd_pool) or sd_pool <= 0:
                continue
            for k, c in enumerate(classes):
                mask = country_idx == k
                if not mask.any():
                    continue
                mu_c_pre  = _weighted_mean(x[mask], w_pre[mask])
                mu_c_post = _weighted_mean(x[mask], w_post[mask])
                smd_pre  = abs(mu_c_pre  - mu_pool) / sd_pool if np.isfinite(mu_c_pre)  else float("nan")
                smd_post = abs(mu_c_post - mu_pool) / sd_pool if np.isfinite(mu_c_post) else float("nan")
                rows.append({
                    "country":  c, "variable": var,
                    "smd_pre":  smd_pre, "smd_post": smd_post,
                    "var_kind": "continuous",
                })
        else:
            sv = s.astype(object).where(s.notna(), "__missing__").astype(str).values
            target_dist = _categorical_dist(sv, w_pre)
            if not target_dist:
                continue
            for k, c in enumerate(classes):
                mask = country_idx == k
                if not mask.any():
                    continue
                pre_dist  = _categorical_dist(sv[mask], w_pre[mask])
                post_dist = _categorical_dist(sv[mask], w_post[mask])
                smd_pre  = _categorical_smd_from_dists(target_dist, pre_dist)
                smd_post = _categorical_smd_from_dists(target_dist, post_dist)
                rows.append({
                    "country":  c, "variable": var,
                    "smd_pre":  smd_pre, "smd_post": smd_post,
                    "var_kind": "categorical",
                })
    return pd.DataFrame(rows)


def balance_per_variable(balance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per SD variable: mean/median/max SMD across the 30 countries, pre vs post.

    Sorted by descending improvement (mean_smd_pre − mean_smd_post).
    """
    rows = []
    for var, sub in balance_df.groupby("variable", sort=False):
        kind = sub["var_kind"].iloc[0] if "var_kind" in sub.columns else ""
        rows.append({
            "variable":                       var,
            "var_kind":                       kind,
            "n_countries":                    int(len(sub)),
            "mean_smd_pre":                   float(sub["smd_pre"].mean()),
            "mean_smd_post":                  float(sub["smd_post"].mean()),
            "median_smd_pre":                 float(sub["smd_pre"].median()),
            "median_smd_post":                float(sub["smd_post"].median()),
            "max_smd_pre":                    float(sub["smd_pre"].max()),
            "max_smd_post":                   float(sub["smd_post"].max()),
            "n_countries_well_balanced_pre":  int((sub["smd_pre"]  < SMD_GOOD).sum()),
            "n_countries_well_balanced_post": int((sub["smd_post"] < SMD_GOOD).sum()),
            "n_countries_imbalanced_pre":     int((sub["smd_pre"]  >= SMD_ACCEPTABLE).sum()),
            "n_countries_imbalanced_post":    int((sub["smd_post"] >= SMD_ACCEPTABLE).sum()),
        })
    out = pd.DataFrame(rows)
    out["improvement"] = out["mean_smd_pre"] - out["mean_smd_post"]
    out = out.sort_values("improvement", ascending=False).reset_index(drop=True)
    return out


def balance_per_country(balance_df: pd.DataFrame, country_name_map: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Per country: mean/max SMD across SD variables, pre vs post.

    Sorted by descending post-IPW mean SMD (worst-balanced countries on top).
    """
    name_map = country_name_map or {}
    rows = []
    for c, sub in balance_df.groupby("country", sort=False):
        rows.append({
            "country":                       c,
            "country_name":                  name_map.get(c, c),
            "n_vars":                        int(len(sub)),
            "mean_smd_pre":                  float(sub["smd_pre"].mean()),
            "mean_smd_post":                 float(sub["smd_post"].mean()),
            "max_smd_pre":                   float(sub["smd_pre"].max()),
            "max_smd_post":                  float(sub["smd_post"].max()),
            "n_vars_well_balanced_pre":      int((sub["smd_pre"]  < SMD_GOOD).sum()),
            "n_vars_well_balanced_post":     int((sub["smd_post"] < SMD_GOOD).sum()),
            "n_vars_imbalanced_pre":         int((sub["smd_pre"]  >= SMD_ACCEPTABLE).sum()),
            "n_vars_imbalanced_post":        int((sub["smd_post"] >= SMD_ACCEPTABLE).sum()),
        })
    out = pd.DataFrame(rows)
    out = out.sort_values("mean_smd_post", ascending=False).reset_index(drop=True)
    return out


def balance_overall(balance_df: pd.DataFrame) -> pd.DataFrame:
    """One-row qset-level balance summary; the headline number is `mean_smd_post`."""
    n_cells = int(len(balance_df))
    row = {
        "n_cells":                       n_cells,
        "mean_smd_pre":                  float(balance_df["smd_pre"].mean())   if n_cells else float("nan"),
        "mean_smd_post":                 float(balance_df["smd_post"].mean())  if n_cells else float("nan"),
        "median_smd_pre":                float(balance_df["smd_pre"].median()) if n_cells else float("nan"),
        "median_smd_post":               float(balance_df["smd_post"].median())if n_cells else float("nan"),
        "max_smd_pre":                   float(balance_df["smd_pre"].max())    if n_cells else float("nan"),
        "max_smd_post":                  float(balance_df["smd_post"].max())   if n_cells else float("nan"),
        "pct_cells_well_balanced_pre":   100.0 * float((balance_df["smd_pre"]  < SMD_GOOD).mean())       if n_cells else float("nan"),
        "pct_cells_well_balanced_post":  100.0 * float((balance_df["smd_post"] < SMD_GOOD).mean())       if n_cells else float("nan"),
        "pct_cells_imbalanced_pre":      100.0 * float((balance_df["smd_pre"]  >= SMD_ACCEPTABLE).mean())if n_cells else float("nan"),
        "pct_cells_imbalanced_post":     100.0 * float((balance_df["smd_post"] >= SMD_ACCEPTABLE).mean())if n_cells else float("nan"),
    }
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# Outcome statistics: country means, between-country variance, R^2
# ─────────────────────────────────────────────────────────────────────────────

def country_weighted_means(
    y: np.ndarray,
    country_idx: np.ndarray,
    classes: list[str],
    w: np.ndarray,
) -> pd.DataFrame:
    """Per-country weighted mean of y, plus n + sum_w."""
    rows = []
    for k, c in enumerate(classes):
        mask = (country_idx == k) & np.isfinite(y) & (w > 0)
        if not mask.any():
            rows.append({"country": c, "mean": float("nan"),
                         "n": 0, "sum_w": 0.0})
            continue
        mean = float(np.average(y[mask], weights=w[mask]))
        rows.append({
            "country": c,
            "mean":    mean,
            "n":       int(mask.sum()),
            "sum_w":   float(w[mask].sum()),
        })
    return pd.DataFrame(rows)


def between_country_variance(
    y: np.ndarray,
    country_idx: np.ndarray,
    classes: list[str],
    w: np.ndarray,
) -> float:
    """
    Population-weighted variance of country-level means.
    Country weights are sum_w per country, normalised to sum to 1.
    """
    K = len(classes)
    means = np.full(K, np.nan)
    sw    = np.zeros(K)
    for k in range(K):
        mask = (country_idx == k) & np.isfinite(y) & (w > 0)
        if not mask.any():
            continue
        means[k] = float(np.average(y[mask], weights=w[mask]))
        sw[k]    = float(w[mask].sum())
    valid = np.isfinite(means) & (sw > 0)
    if not valid.any():
        return float("nan")
    pw = sw[valid] / sw[valid].sum()
    mu = float((pw * means[valid]).sum())
    var = float((pw * (means[valid] - mu) ** 2).sum())
    return var


# ─────────────────────────────────────────────────────────────────────────────
# Point-estimate runner: full pipeline on one sample
# ─────────────────────────────────────────────────────────────────────────────

def _setting_country_means(
    dev: np.ndarray,
    country_idx: np.ndarray,
    classes: list[str],
    pspwght: np.ndarray,
    fw: np.ndarray,
) -> dict:
    """
    Two country-mean series for one final-weight vector:

    * ``country_means_reweighted``        — IPW-final-weighted mean per country.
    * ``country_means_pspwght_survivors`` — ``pspwght``-weighted mean on the
      SURVIVING subset (``fw > 0``).  Equals the design-weighted full-sample
      mean for clip settings (no drops).  For drop settings it isolates the
      population-restriction effect from the reweighting effect.
    """
    pspwght_surv = np.where(fw > 0, pspwght, 0.0)
    return {
        "country_means_reweighted":
            country_weighted_means(dev, country_idx, classes, fw),
        "country_means_pspwght_survivors":
            country_weighted_means(dev, country_idx, classes, pspwght_surv),
    }


def _headline_block(
    dev: np.ndarray,
    country_idx: np.ndarray,
    classes: list[str],
    pspwght: np.ndarray,
    fw: np.ndarray,
    per_model_y: dict[str, np.ndarray],
    seed: int,
    new_id: np.ndarray | None = None,
) -> dict:
    """
    Headline between-country variance metrics for one final-weight vector.

    Variance flavours emitted:

    * ``pre``       : ``pspwght`` on the full sample (same across settings).
    * ``post``      : final weight ``fw`` directly.  For drop settings this
                      restricts to the surviving sample (``fw > 0``).
    * ``post_full`` : ``where(fw>0, fw, pspwght)``.  Dropped respondents fall
                      back to design weight, so the metric is always on the
                      full 50,115 regardless of setting.  Used by
                      ``settings_summary.csv`` as the option-2 robustness
                      framing for drop settings; equals ``post`` for clip.

    Country-mean series (two): see ``_setting_country_means``.
    """
    fw_for_full  = np.where(fw > 0, fw, pspwght)

    var_pre       = between_country_variance(dev, country_idx, classes, pspwght)
    var_post      = between_country_variance(dev, country_idx, classes, fw)
    var_post_full = between_country_variance(dev, country_idx, classes, fw_for_full)
    ratio      = (var_post      / var_pre) if var_pre > 0 else float("nan")
    ratio_full = (var_post_full / var_pre) if var_pre > 0 else float("nan")

    per_model: dict[str, dict] = {}
    for ms, a in per_model_y.items():
        v_pre_m       = between_country_variance(a, country_idx, classes, pspwght)
        v_post_m      = between_country_variance(a, country_idx, classes, fw)
        v_post_full_m = between_country_variance(a, country_idx, classes, fw_for_full)
        per_model[ms] = {
            "variance_pre":        v_pre_m,
            "variance_post":       v_post_m,
            "variance_post_full":  v_post_full_m,
            "variance_ratio":      (v_post_m      / v_pre_m) if v_pre_m > 0 else float("nan"),
            "variance_ratio_full": (v_post_full_m / v_pre_m) if v_pre_m > 0 else float("nan"),
        }

    cm = _setting_country_means(dev, country_idx, classes, pspwght, fw)
    return {
        "variance": {
            "pre":        var_pre,
            "post":       var_post,
            "post_full":  var_post_full,
            "ratio":      ratio,
            "ratio_full": ratio_full,
        },
        # Alias for the reweighted series, so callers that only need the
        # primary country means do not have to pick a series name.
        "country_means":                   cm["country_means_reweighted"],
        "country_means_reweighted":        cm["country_means_reweighted"],
        "country_means_pspwght_survivors": cm["country_means_pspwght_survivors"],
        "per_model":                       per_model,
    }


def _evaluate_qset(
    qset_df: pd.DataFrame,
    qset_oof: np.ndarray,
    model_shorts: list[str],
    classes: list[str],
    seed: int,
    means_only: bool = False,
) -> dict:
    """
    Given a qset's df + the corresponding OOF propensity rows (already
    extracted from a possibly-joint propensity fit), compute IPW weights and
    headline metrics for every setting in ``SETTING_DEFS``.

    The OOF column order must match ``classes`` (i.e. the union's class set,
    which the qset's countries must be a subset of).  In the typical case
    qset classes == union classes.

    ``means_only=True`` skips the headline variance computations and
    returns only the two country-mean series per setting — used inside
    the bootstrap when no per-LLM σ is needed.  The bootstrap actually
    calls ``_evaluate_qset(means_only=False)`` so we can pick up per-LLM
    between-country variance per replicate for the CIs on Δσ.
    """
    country_idx, classes_q = _country_codes(qset_df)
    if classes_q != classes:
        # Permissive fallback: use qset's own class set if it differs.  In the
        # current setup every qset has the same 30 countries, so this branch
        # never fires.
        classes = classes_q
    K = len(classes)
    pspwght = qset_df["pspwght"].values.astype(float)
    new_id  = qset_df["new_id"].values

    dev_series, ref_means = compute_cross_model_deviation(qset_df, model_shorts)
    dev = dev_series.values.astype(float)

    ipw_weight, final_weight_raw, marginals = compute_raw_ipw_weights(
        country_idx, qset_oof, pspwght,
    )
    N = len(qset_df)
    prop_own     = qset_oof[np.arange(N), country_idx]
    own_marginal = marginals[country_idx]

    weights_by_setting, affected_by_setting = _compute_setting_weights(
        prop_own, own_marginal, pspwght, country_idx, K,
    )

    per_model_y: dict[str, np.ndarray] = {
        ms: qset_df[f"align_{ms}"].values.astype(float) for ms in model_shorts
    }

    out: dict = {
        "df":                qset_df,
        "model_shorts":      model_shorts,
        "country_idx":       country_idx,
        "classes":           classes,
        "pspwght":           pspwght,
        "oof_proba":         qset_oof,
        "prop_own":          prop_own,
        "dev":               dev,
        "ref_means":         ref_means,
        "ipw_weight":        ipw_weight,
        "final_weight_raw":  final_weight_raw,
        "country_marginal":  marginals,
        "per_model_y":       per_model_y,
        "weights":           weights_by_setting,   # dict[setting -> (N,) fw]
        "affected":          affected_by_setting,  # dict[setting -> (N,) bool]
    }
    for setting in SETTING_NAMES:
        fw = weights_by_setting[setting]
        if means_only:
            cm = _setting_country_means(dev, country_idx, classes, pspwght, fw)
            out[setting] = {
                "country_means":                   cm["country_means_reweighted"],
                "country_means_reweighted":        cm["country_means_reweighted"],
                "country_means_pspwght_survivors": cm["country_means_pspwght_survivors"],
            }
        else:
            out[setting] = _headline_block(
                dev, country_idx, classes, pspwght, fw, per_model_y,
                seed=seed, new_id=new_id,
            )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_one(
    qset_df: pd.DataFrame,
    model_shorts: list[str],
    iter_seed: int,
    crossfit_folds: int,
    propensity_params: dict,
) -> dict:
    """
    One bootstrap replicate.

    Resamples the qset within country, fits cross-fitted propensity once,
    and collects:

      * per-country cross-model means under each (setting, series) for the
        country_means CIs;
      * per-LLM between-country σ under ``pspwght`` (σ_pre) and under each
        setting's IPW weights (σ_post) — paired per replicate so
        Δσ = σ_post − σ_pre stays well-defined inside the replicate.

    Both σ's use the within-survivors framing for drop settings (option 1),
    matching the variance tables.
    """
    rng = np.random.RandomState(iter_seed)
    country_idx, classes = _country_codes(qset_df)

    # Stratified resample within country
    idx_parts = []
    for k in range(len(classes)):
        in_k = np.where(country_idx == k)[0]
        if len(in_k) == 0:
            continue
        sel = rng.randint(0, len(in_k), size=len(in_k))
        idx_parts.append(in_k[sel])
    boot_idx = np.concatenate(idx_parts)
    qset_boot = qset_df.iloc[boot_idx].reset_index(drop=True)

    qset_oof_boot, classes_boot, _meta = fit_crossfitted_propensity(
        qset_boot,
        n_splits=crossfit_folds,
        params=propensity_params,
        seed=iter_seed,
    )

    op_by_setting = {name: op for name, op, _ in SETTING_DEFS}

    eval_res = _evaluate_qset(
        qset_boot, qset_oof_boot, model_shorts, classes_boot,
        seed=iter_seed, means_only=True,
    )

    cm_parts = []
    for setting in SETTING_NAMES:
        cm_rew = eval_res[setting]["country_means_reweighted"][["country", "mean"]].rename(
            columns={"mean": "country_mean_post"}
        )
        cm_rew["setting"] = setting
        cm_rew["series"]  = "reweighted"
        cm_parts.append(cm_rew)
        if op_by_setting[setting] == "drop":
            cm_pw = eval_res[setting]["country_means_pspwght_survivors"][["country", "mean"]].rename(
                columns={"mean": "country_mean_post"}
            )
            cm_pw["setting"] = setting
            cm_pw["series"]  = "pspwght_survivors"
            cm_parts.append(cm_pw)

    country_idx_q = eval_res["country_idx"]
    classes_q     = eval_res["classes"]
    pspwght_q     = eval_res["pspwght"]
    per_model_y   = eval_res["per_model_y"]
    weights_by    = eval_res["weights"]
    sigma_rows: list[dict] = []
    for ms, y in per_model_y.items():
        var_pre = between_country_variance(y, country_idx_q, classes_q, pspwght_q)
        sigma_pre = float(np.sqrt(max(var_pre, 0.0)))
        for setting in SETTING_NAMES:
            fw = weights_by[setting]
            var_post = between_country_variance(y, country_idx_q, classes_q, fw)
            sigma_post = float(np.sqrt(max(var_post, 0.0)))
            sigma_rows.append({
                "iter_seed":  int(iter_seed),
                "llm_model":  ms,
                "setting":    setting,
                "sigma_pre":  sigma_pre,
                "sigma_post": sigma_post,
            })

    return {
        "iter_seed":      int(iter_seed),
        "_country_means": pd.concat(cm_parts, ignore_index=True).to_dict(orient="records"),
        "_sigma":         sigma_rows,
    }


def run_bootstrap(
    qset_df: pd.DataFrame,
    model_shorts: list[str],
    n_replicates: int,
    crossfit_folds: int,
    propensity_params: dict,
    base_seed: int = RANDOM_SEED,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parallel bootstrap.  Each replicate fits one propensity on a stratified
    resample and emits per-country cross-model means + per-LLM σ_pre/σ_post.

    Returns a triple (replicate_results, country_means, per_llm_sigma).
    """
    seeds = [base_seed + 1 + i for i in range(n_replicates)]
    t0 = time.time()

    def _wrap(s, idx):
        r = _bootstrap_one(
            qset_df, model_shorts, s, crossfit_folds, propensity_params,
        )
        if (idx + 1) % max(1, n_replicates // 20) == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-6)
            eta  = (n_replicates - idx - 1) / max(rate, 1e-6)
            print(f"      bootstrap {idx + 1}/{n_replicates}  "
                  f"({elapsed/60:.1f}m elapsed, {eta/60:.1f}m ETA)")
        return r

    results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=1)(
        delayed(_wrap)(s, i) for i, s in enumerate(seeds)
    )

    rep_rows: list[dict] = []
    cm_rows:  list[dict] = []
    sig_rows: list[dict] = []
    for rep in results:
        cms = rep.pop("_country_means")
        sigs = rep.pop("_sigma")
        rep_rows.append(rep)
        for cm in cms:
            cm_rows.append({"iter_seed": rep["iter_seed"], **cm})
        for sig in sigs:
            sig_rows.append(sig)

    return (pd.DataFrame(rep_rows),
            pd.DataFrame(cm_rows),
            pd.DataFrame(sig_rows))


def summarise_bootstrap(
    df_cm: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Percentile-method CIs for the per-country means across (setting, series).

    ``df_cm`` is the long-format table of per-replicate means with columns
    (iter_seed, country, country_mean_post, setting, series).

    Returns a dict with a single entry ``country_mean_ci`` (DataFrame with
    columns country, setting, series, boot_mean, boot_sd, ci_lo, ci_hi,
    n_replicates).  Per-LLM σ CIs are produced separately by
    ``summarise_sigma_diff`` from the bootstrap's ``_sigma`` channel.
    """
    cm_rows: list[dict] = []
    has_setting = "setting" in df_cm.columns
    has_series  = "series"  in df_cm.columns
    group_keys = ["country"]
    if has_setting: group_keys.append("setting")
    if has_series:  group_keys.append("series")

    for keys, sub in df_cm.groupby(group_keys, sort=True):
        vals = sub["country_mean_post"].dropna().values
        if len(vals) < 2:
            continue
        lo, hi = np.quantile(vals, [0.025, 0.975])
        rec: dict = {
            "country":      keys[0] if isinstance(keys, tuple) else keys,
            "boot_mean":    float(np.mean(vals)),
            "boot_sd":      float(np.std(vals)),
            "ci_lo":        float(lo),
            "ci_hi":        float(hi),
            "n_replicates": int(len(vals)),
        }
        if has_setting:
            rec["setting"] = keys[1]
        if has_series:
            rec["series"]  = keys[2] if has_setting else keys[1]
        cm_rows.append(rec)
    return {"country_mean_ci": pd.DataFrame(cm_rows)}


# ─────────────────────────────────────────────────────────────────────────────
# Variance tables (paper-ready CSV + TeX)
# ─────────────────────────────────────────────────────────────────────────────

def summarise_sigma_diff(df_boot_sigma: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile-method 95% CI on Δσ = σ_post − σ_pre per (llm_model, setting).

    Input columns: iter_seed, llm_model, setting, sigma_pre, sigma_post.
    Output columns: llm_model, setting, boot_mean_diff, boot_sd_diff,
    ci_lo, ci_hi, n_replicates.
    """
    rows: list[dict] = []
    for keys, sub in df_boot_sigma.groupby(["llm_model", "setting"], sort=True):
        diff = (sub["sigma_post"] - sub["sigma_pre"]).dropna().values
        if len(diff) < 2:
            continue
        lo, hi = np.quantile(diff, [0.025, 0.975])
        rows.append({
            "llm_model":      keys[0],
            "setting":        keys[1],
            "boot_mean_diff": float(np.mean(diff)),
            "boot_sd_diff":   float(np.std(diff)),
            "ci_lo":          float(lo),
            "ci_hi":          float(hi),
            "n_replicates":   int(len(diff)),
        })
    return pd.DataFrame(rows)


def _write_variance_tables(
    per_model_df: pd.DataFrame,
    sigma_diff_ci: pd.DataFrame | None,
    outs_dir: Path,
) -> None:
    """
    Write two paper-ready per-LLM variance tables (CSV + TeX):

    - ``variance_summary_clip.{csv,tex}`` (main analysis):
        columns σ_pre, σ_clip, Δ = σ_clip − σ_pre, [CI_lo, CI_hi] on Δ.
    - ``variance_summary_drop.{csv,tex}`` (robustness):
        σ_pre on the left, then (σ_post, Δ, CI) for drop_p01 and drop_p05.

    σ values come from the point-estimate ``per_model_df`` (square roots of
    ``variance_pre`` / ``variance_post``).  CI on Δ comes from the per-replicate
    bootstrap (``sigma_diff_ci``).  Both tables use within-survivors framing
    (option 1) to match the cross-model-deviation plot.
    """
    # ── σ point estimates per (llm, setting) ────────────────────────────────
    primary = per_model_df[per_model_df["setting"] == PRIMARY_SETTING].set_index("llm_model")
    sigma_pre = primary["variance_pre"].clip(lower=0).pow(0.5)

    def _sigma_post(setting: str) -> pd.Series:
        sub = per_model_df[per_model_df["setting"] == setting].set_index("llm_model")
        return sub["variance_post"].clip(lower=0).pow(0.5)

    sigma_clip = _sigma_post("clip_p01")
    sigma_dp01 = _sigma_post("drop_p01")
    sigma_dp05 = _sigma_post("drop_p05")

    # ── CI lookups: (setting, llm) -> (ci_lo, ci_hi, boot_mean_diff) ────────
    def _ci(setting: str, llm: str) -> tuple[float, float]:
        if sigma_diff_ci is None or sigma_diff_ci.empty:
            return float("nan"), float("nan")
        m = sigma_diff_ci[(sigma_diff_ci["setting"] == setting)
                          & (sigma_diff_ci["llm_model"] == llm)]
        if m.empty:
            return float("nan"), float("nan")
        r = m.iloc[0]
        return float(r["ci_lo"]), float(r["ci_hi"])

    def _fmt_ci(lo: float, hi: float) -> str:
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return "--"
        return f"[{lo:+.4f}, {hi:+.4f}]"

    llms = sorted(primary.index.tolist())

    # ── Clip table ──────────────────────────────────────────────────────────
    clip_rows = []
    for llm in llms:
        sp = float(sigma_pre.get(llm, float("nan")))
        sc = float(sigma_clip.get(llm, float("nan")))
        diff = sc - sp
        lo, hi = _ci("clip_p01", llm)
        clip_rows.append({
            "llm_model": llm,
            "sigma_pre": sp,
            "sigma_clip": sc,
            "delta":     diff,
            "ci_lo":     lo,
            "ci_hi":     hi,
        })
    clip_df = pd.DataFrame(clip_rows)
    clip_df.to_csv(outs_dir / "variance_summary_clip.csv", index=False)

    clip_tex_df = pd.DataFrame({
        "LLM":             clip_df["llm_model"],
        r"$\sigma_{pre}$": clip_df["sigma_pre"].map(lambda v: f"{v:.4f}"),
        r"$\sigma_{clip}$": clip_df["sigma_clip"].map(lambda v: f"{v:.4f}"),
        r"$\Delta$":       clip_df["delta"].map(lambda v: f"{v:+.4f}"),
        r"95\% CI on $\Delta$": [_fmt_ci(lo, hi) for lo, hi
                                  in zip(clip_df["ci_lo"], clip_df["ci_hi"])],
    })
    try:
        tex = clip_tex_df.to_latex(
            index=False,
            column_format="lrrrr",
            caption=("Between-country standard deviation of cross-model alignment "
                     "deviation per LLM, before and after clip-IPW reweighting "
                     "(clip @ $\\hat{P}<0.01$, main analysis).  $\\Delta = "
                     "\\sigma_{clip}-\\sigma_{pre}$; 95\\% bootstrap CI."),
            label="tab:ipw_variance_clip",
            escape=False,
        )
        with open(outs_dir / "variance_summary_clip.tex", "w", encoding="utf-8") as fh:
            fh.write(tex)
    except Exception as e:
        warnings.warn(f"variance_summary_clip.tex export failed: {e}")

    # ── Drop robustness table ───────────────────────────────────────────────
    drop_rows = []
    for llm in llms:
        sp = float(sigma_pre.get(llm, float("nan")))
        s1 = float(sigma_dp01.get(llm, float("nan")))
        s5 = float(sigma_dp05.get(llm, float("nan")))
        d1 = s1 - sp; d5 = s5 - sp
        lo1, hi1 = _ci("drop_p01", llm)
        lo5, hi5 = _ci("drop_p05", llm)
        drop_rows.append({
            "llm_model":        llm,
            "sigma_pre":        sp,
            "sigma_drop_p01":   s1,
            "delta_drop_p01":   d1,
            "ci_lo_drop_p01":   lo1,
            "ci_hi_drop_p01":   hi1,
            "sigma_drop_p05":   s5,
            "delta_drop_p05":   d5,
            "ci_lo_drop_p05":   lo5,
            "ci_hi_drop_p05":   hi5,
        })
    drop_df = pd.DataFrame(drop_rows)
    drop_df.to_csv(outs_dir / "variance_summary_drop.csv", index=False)

    drop_tex_df = pd.DataFrame({
        "LLM":             drop_df["llm_model"],
        r"$\sigma_{pre}$": drop_df["sigma_pre"].map(lambda v: f"{v:.4f}"),
        r"$\sigma_{drop_{p01}}$": drop_df["sigma_drop_p01"].map(lambda v: f"{v:.4f}"),
        r"$\Delta_{p01}$":        drop_df["delta_drop_p01"].map(lambda v: f"{v:+.4f}"),
        r"95\% CI ($p01$)":       [_fmt_ci(lo, hi) for lo, hi
                                    in zip(drop_df["ci_lo_drop_p01"], drop_df["ci_hi_drop_p01"])],
        r"$\sigma_{drop_{p05}}$": drop_df["sigma_drop_p05"].map(lambda v: f"{v:.4f}"),
        r"$\Delta_{p05}$":        drop_df["delta_drop_p05"].map(lambda v: f"{v:+.4f}"),
        r"95\% CI ($p05$)":       [_fmt_ci(lo, hi) for lo, hi
                                    in zip(drop_df["ci_lo_drop_p05"], drop_df["ci_hi_drop_p05"])],
    })
    try:
        tex = drop_tex_df.to_latex(
            index=False,
            column_format="lrrrrrrr",
            caption=("Between-country standard deviation of cross-model alignment "
                     "deviation per LLM under the drop-IPW robustness checks "
                     "(drop @ $\\hat{P}<0.01$ and $\\hat{P}<0.05$).  "
                     "$\\Delta = \\sigma_{drop}-\\sigma_{pre}$; 95\\% bootstrap CI."),
            label="tab:ipw_variance_drop",
            escape=False,
        )
        with open(outs_dir / "variance_summary_drop.tex", "w", encoding="utf-8") as fh:
            fh.write(tex)
    except Exception as e:
        warnings.warn(f"variance_summary_drop.tex export failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def _write_qset_artifacts(
    output_dir: Path,
    qset_key: str,
    eval_res: dict,
    model_shorts: list[str],
    classes: list[str],
    weight_col: str,
    seed: int,
    union_meta: dict,
    sd_present: list[str],
    n_bootstrap: int,
    crossfit_folds: int,
    df_boot_res: pd.DataFrame | None = None,
    df_boot_cm: pd.DataFrame | None = None,
    df_boot_sigma: pd.DataFrame | None = None,
) -> None:
    """
    Write the full per-qset artifact tree given an already-computed
    ``eval_res`` from ``_evaluate_qset`` and (optionally) bootstrap replicate
    DataFrames.

    Called by ``run_ipw`` twice: first with no bootstrap inputs to emit
    point estimates, then a second time (if bootstrap runs) to overwrite
    the variance tables / country-means plots with CI overlays.
    """
    from .ipw_plots import (
        plot_propensity_histograms,
        plot_balance_per_variable, plot_balance_per_country,
        plot_country_means_three_stage,
        plot_cross_model_deviation_before_after,
    )
    from .config import COUNTRY_NAME

    output_dir = Path(output_dir)
    diag_dir   = output_dir / "diagnostics"
    boot_dir   = output_dir / "bootstrap"
    outs_dir   = output_dir / "outputs"
    for d in (diag_dir, outs_dir):
        d.mkdir(parents=True, exist_ok=True)

    df          = eval_res["df"]
    country_idx = eval_res["country_idx"]
    K           = len(classes)
    pspwght     = eval_res["pspwght"]
    new_id      = df["new_id"].values
    proba       = eval_res["oof_proba"]
    prop_own    = eval_res["prop_own"]
    dev         = eval_res["dev"]
    fw_raw      = eval_res["final_weight_raw"]
    weights_by  = eval_res["weights"]
    affected_by = eval_res["affected"]

    # ── Artefact: propensity_oof.csv.gz ─────────────────────────────────────
    print("  [2/5] Saving propensity + weights artefacts …")
    prop_df = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])
    prop_df.insert(0, "new_id", new_id)
    prop_df["argmax_country"] = [classes[i] for i in np.argmax(proba, axis=1)]
    prop_df.to_csv(output_dir / "propensity_oof.csv.gz", index=False)

    # ── Artefact: weights.csv.gz (all settings side-by-side) ────────────────
    weights_cols = {
        "new_id":           new_id,
        "cntry":            df["cntry"].values,
        "pspwght":          pspwght,
        "prop_own":         prop_own,
        "ipw_weight":       eval_res["ipw_weight"],
        "final_weight_raw": fw_raw,
    }
    for setting in SETTING_NAMES:
        weights_cols[f"final_weight_{setting}"] = weights_by[setting]
        weights_cols[f"affected_{setting}"]     = affected_by[setting]
    pd.DataFrame(weights_cols).to_csv(output_dir / "weights.csv.gz", index=False)

    # ── Artefact: propensity_model_meta.json ────────────────────────────────
    meta = dict(union_meta)
    meta["model_shorts"]       = model_shorts
    meta["question_set_key"]   = qset_key
    meta["countries"]          = classes
    meta["n_respondents"]      = int(len(df))
    meta["sd_covariates_used"] = sd_present
    meta["weight_col"]         = weight_col
    meta["timestamp"]          = datetime.now().isoformat()
    meta["ref_means"]          = eval_res["ref_means"]
    with open(output_dir / "propensity_model_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=float)

    # ── Artefact: weight_summary.csv (per country × setting) ────────────────
    # Op-type per setting (clip vs drop) for the n_affected accounting.
    op_by_setting = {name: op for name, op, _ in SETTING_DEFS}

    win_rows: list[dict] = []
    ess_pre_arr = kish_ess_per_country(pspwght, country_idx, K)
    for setting in SETTING_NAMES:
        fw       = weights_by[setting]
        affected = affected_by[setting]
        op       = op_by_setting[setting]
        ess_post = kish_ess_per_country(fw, country_idx, K)
        for k, c in enumerate(classes):
            mask = country_idx == k
            n_k = int(mask.sum())
            if n_k == 0:
                continue
            n_affected = int(affected[mask].sum())
            n_clipped = n_affected if op == "clip" else 0
            n_dropped = n_affected if op == "drop" else 0
            loss_pct = (100.0 * (ess_pre_arr[k] - ess_post[k]) / ess_pre_arr[k]
                        if ess_pre_arr[k] > 0 else float("nan"))
            win_rows.append({
                "setting":           setting,
                "country":           c,
                "n":                 n_k,
                "n_affected":        n_affected,
                "n_clipped":         n_clipped,
                "n_dropped":         n_dropped,
                "max_weight_pre":    float(fw_raw[mask].max()),
                "max_weight_post":   float(fw[mask].max()),
                "kish_ess_pre":      float(ess_pre_arr[k]),
                "kish_ess_post":     float(ess_post[k]),
                "kish_ess_loss_pct": float(loss_pct),
            })
    pd.DataFrame(win_rows).to_csv(output_dir / "weight_summary.csv", index=False)

    # ── Diagnostics ─────────────────────────────────────────────────────────
    overall_metrics = propensity_metrics_overall(proba, country_idx, classes, pspwght)
    pd.DataFrame([overall_metrics]).to_csv(
        diag_dir / "propensity_metrics.csv", index=False,
    )
    plot_propensity_histograms(
        proba, country_idx, classes,
        diag_dir / "propensity_histograms.png", qset_key,
    )

    # Balance vs the pooled-European target — pre uses pspwght, post uses the
    # primary setting (clip_p01).  Drop settings are reported alongside in the
    # headline tables but the balance check is on the primary.
    balance_df = compute_balance_table(
        df, country_idx, classes, pspwght, weights_by[PRIMARY_SETTING],
    )
    balance_df.to_csv(diag_dir / "balance_smd.csv", index=False)

    bal_var     = balance_per_variable(balance_df)
    bal_country = balance_per_country(balance_df, country_name_map=COUNTRY_NAME)
    bal_overall = balance_overall(balance_df)
    bal_var.to_csv(diag_dir / "balance_per_variable.csv", index=False)
    bal_country.to_csv(diag_dir / "balance_per_country.csv", index=False)
    bal_overall.to_csv(diag_dir / "balance_overall.csv", index=False)

    plot_balance_per_variable(
        bal_var, diag_dir / "balance_per_variable.png", qset_key,
    )
    plot_balance_per_country(
        bal_country, diag_dir / "balance_per_country.png", qset_key,
    )

    # ── Outputs (paper) ─────────────────────────────────────────────────────
    var_rows: list[dict] = []
    per_model_rows: list[dict] = []

    for setting in SETTING_NAMES:
        b = eval_res[setting]
        var_rows.append({
            "setting":              setting,
            "variance_pre":         b["variance"]["pre"],
            "variance_post":        b["variance"]["post"],
            "variance_post_full":   b["variance"]["post_full"],
            "variance_ratio":       b["variance"]["ratio"],
            "variance_ratio_full":  b["variance"]["ratio_full"],
        })
        for ms, m in b["per_model"].items():
            per_model_rows.append({
                "setting":             setting,
                "llm_model":           ms,
                "variance_pre":        m["variance_pre"],
                "variance_post":       m["variance_post"],
                "variance_post_full":  m["variance_post_full"],
                "variance_ratio":      m["variance_ratio"],
                "variance_ratio_full": m["variance_ratio_full"],
            })

    per_model_df = pd.DataFrame(per_model_rows)
    per_model_df.to_csv(outs_dir / "per_model_summary.csv", index=False)

    # ── Paper-ready settings_summary.csv (3 rows; variance columns only) ────
    win_df_local = pd.DataFrame(win_rows)
    cd_summary_rows = []
    for v_row in var_rows:
        setting = v_row["setting"]
        affected = int(
            win_df_local.loc[win_df_local["setting"] == setting,
                             "n_affected"].sum()
        )
        cd_summary_rows.append({
            "setting":              setting,
            "op":                   op_by_setting[setting],
            "variance_pre":         v_row["variance_pre"],
            "variance_post":        v_row["variance_post"],
            "variance_post_full":   v_row["variance_post_full"],
            "variance_ratio":       v_row["variance_ratio"],
            "variance_ratio_full":  v_row["variance_ratio_full"],
            "n_affected":           affected,
        })
    pd.DataFrame(cd_summary_rows).to_csv(outs_dir / "settings_summary.csv", index=False)

    # ── Bootstrap artefacts (country means + per-LLM σ) ─────────────────────
    country_mean_ci_df = None
    have_boot = (n_bootstrap > 0
                 and df_boot_res is not None
                 and df_boot_cm is not None)
    if have_boot or df_boot_sigma is not None:
        boot_dir.mkdir(parents=True, exist_ok=True)
    if have_boot:
        df_boot_res.to_csv(boot_dir / "replicate_results.csv.gz", index=False)
        df_boot_cm.to_csv(boot_dir  / "country_means_replicates.csv.gz", index=False)

        boot_summ = summarise_bootstrap(df_boot_cm)
        boot_summ["country_mean_ci"].to_csv(boot_dir / "country_mean_ci.csv", index=False)
        country_mean_ci_df = boot_summ["country_mean_ci"]
    if df_boot_sigma is not None and not df_boot_sigma.empty:
        df_boot_sigma.to_csv(boot_dir / "sigma_replicates.csv.gz", index=False)

    # ── Artefacts for plot regeneration ─────────────────────────────────────
    # (1) Per-respondent cross-model mean deviation + country index
    pd.DataFrame({
        "new_id":      new_id,
        "country_idx": country_idx,
        "dev":         dev,
    }).to_csv(outs_dir / "respondent_dev.csv.gz", index=False)

    # (2) Country-level point-estimate means per setting × series
    cm_pe_rows: list[dict] = []
    for setting in SETTING_NAMES:
        for row in eval_res[setting]["country_means_reweighted"].itertuples(index=False):
            cm_pe_rows.append({"setting": setting, "series": "reweighted",
                               "country": row.country, "mean": row.mean})
        surv = eval_res[setting]["country_means_pspwght_survivors"]
        if surv is not None and not surv.empty:
            for row in surv.itertuples(index=False):
                cm_pe_rows.append({"setting": setting, "series": "pspwght_survivors",
                                   "country": row.country, "mean": row.mean})
    pd.DataFrame(cm_pe_rows).to_csv(outs_dir / "country_means_point_estimates.csv", index=False)

    # (3) Per-LLM per-country deviations pre/post (for before-after plot)
    _fw_clip_tmp = weights_by[PRIMARY_SETTING]
    _per_model_y_tmp = eval_res["per_model_y"]
    _ref_means_tmp   = eval_res["ref_means"]

    def _per_country_per_llm_dev_tmp(weight_vec: np.ndarray) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {c: {} for c in classes}
        for ms in model_shorts:
            y = _per_model_y_tmp[ms]
            llm_overall = _ref_means_tmp.get(ms)
            if llm_overall is None or not np.isfinite(llm_overall):
                continue
            for k, c in enumerate(classes):
                mask = (country_idx == k) & np.isfinite(y) & (weight_vec > 0)
                if mask.any():
                    cm = float(np.average(y[mask], weights=weight_vec[mask]))
                    out[c][ms] = cm - float(llm_overall)
        return out

    _dev_pre_tmp  = _per_country_per_llm_dev_tmp(pspwght)
    _dev_post_tmp = _per_country_per_llm_dev_tmp(_fw_clip_tmp)
    llm_dev_rows: list[dict] = []
    for stage, d in [("pre", _dev_pre_tmp), ("post", _dev_post_tmp)]:
        for country, llm_vals in d.items():
            for ms, val in llm_vals.items():
                llm_dev_rows.append({"stage": stage, "country": country,
                                     "llm_model": ms, "deviation": val})
    pd.DataFrame(llm_dev_rows).to_csv(outs_dir / "per_llm_country_deviations.csv", index=False)

    # ── Figures ─────────────────────────────────────────────────────────────
    # One country-means forest per setting; primary (clip_p01) has no suffix.
    # For drop settings, a third "overlap means without reweighting" row is
    # drawn (pspwght-weighted means restricted to the surviving subset).
    for setting in SETTING_NAMES:
        fw = weights_by[setting]
        cm_ci_rew    = None
        cm_ci_pwsurv = None
        if country_mean_ci_df is not None and "setting" in country_mean_ci_df.columns:
            sub = country_mean_ci_df[country_mean_ci_df["setting"] == setting]
            if "series" in sub.columns:
                cm_ci_rew    = sub[sub["series"] == "reweighted"       ].drop(columns=["setting", "series"]).reset_index(drop=True)
                cm_ci_pwsurv = sub[sub["series"] == "pspwght_survivors"].drop(columns=["setting", "series"]).reset_index(drop=True)
                if cm_ci_pwsurv.empty:
                    cm_ci_pwsurv = None
            else:
                cm_ci_rew = sub.drop(columns=["setting"]).reset_index(drop=True)
        suffix = "" if setting == PRIMARY_SETTING else f"_{setting}"
        op = op_by_setting[setting]
        pspwght_survivors_means = (
            eval_res[setting]["country_means_pspwght_survivors"]
            if op == "drop" else None
        )
        plot_country_means_three_stage(
            df_country=eval_res[setting]["country_means_reweighted"],
            country_mean_ci=cm_ci_rew,
            df=df, country_idx=country_idx, classes=classes,
            dev=dev, pspwght=pspwght,
            fw_primary=fw,
            output_path=outs_dir / f"country_means_three_stage{suffix}.png",
            question_set_key=qset_key,
            pspwght_survivors_means=pspwght_survivors_means,
            pspwght_survivors_ci=cm_ci_pwsurv,
        )

    # ── Paper-ready variance tables (clip + drop) with bootstrap CIs on Δσ ──
    sigma_diff_ci_df = None
    if df_boot_sigma is not None and not df_boot_sigma.empty:
        sigma_diff_ci_df = summarise_sigma_diff(df_boot_sigma)
        sigma_diff_ci_df.to_csv(boot_dir / "sigma_diff_ci.csv", index=False)
    _write_variance_tables(per_model_df, sigma_diff_ci_df, outs_dir)

    # ── Clip-only cross-model deviation "before-after" plot ─────────────────
    # Use the pre-computed per-LLM deviations saved to CSV above.
    cm_pre_df = pd.DataFrame({
        "country": classes,
        "mean": [float(np.mean(list(_dev_pre_tmp.get(c, {}).values()) or [np.nan]))
                 for c in classes],
    })
    cm_post_df = eval_res[PRIMARY_SETTING]["country_means_reweighted"][["country", "mean"]].reset_index(drop=True)
    plot_cross_model_deviation_before_after(
        classes=classes,
        per_country_cross_model_dev_pre=cm_pre_df,
        per_country_cross_model_dev_post=cm_post_df,
        per_llm_country_dev_pre=_dev_pre_tmp,
        per_llm_country_dev_post=_dev_post_tmp,
        output_path=outs_dir / "cross_model_deviation_before_after_clip_p01.png",
        question_set_key=qset_key,
    )

    # ── run_metadata.json ───────────────────────────────────────────────────
    runmeta = {
        "question_set_key":   qset_key,
        "weight_col":         weight_col,
        "model_shorts":       model_shorts,
        "classes":            list(classes),
        "n_respondents":      int(len(df)),
        "n_countries":        len(classes),
        "sd_covariates_used": sd_present,
        "crossfit_folds":     crossfit_folds,
        "n_bootstrap":        int(n_bootstrap),
        "seed":               seed,
        "timestamp":          datetime.now().isoformat(),
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(runmeta, fh, indent=2)
    print(f"    artefacts written to {output_dir}")


def regen_ipw_plots(output_dir: Path) -> None:
    """
    Regenerate all IPW figures from the CSVs saved by a previous ``run_ipw``
    call.  Does not re-fit any models.

    Reads from:
      outputs/respondent_dev.csv.gz
      outputs/country_means_point_estimates.csv
      outputs/per_llm_country_deviations.csv
      weights.csv.gz
      diagnostics/balance_per_variable.csv
      diagnostics/balance_per_country.csv
      propensity_oof.csv.gz
      bootstrap/country_mean_ci.csv          (optional)
      run_metadata.json
    """
    from .ipw_plots import (
        plot_propensity_histograms,
        plot_balance_per_variable, plot_balance_per_country,
        plot_country_means_three_stage,
        plot_cross_model_deviation_before_after,
    )

    output_dir = Path(output_dir)
    diag_dir   = output_dir / "diagnostics"
    boot_dir   = output_dir / "bootstrap"
    outs_dir   = output_dir / "outputs"

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta_path = output_dir / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_metadata.json not found in {output_dir}")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    qset_key     = meta["question_set_key"]
    classes      = meta["classes"]
    model_shorts = meta["model_shorts"]

    # ── Respondent-level dev + country_idx ───────────────────────────────────
    dev_df      = pd.read_csv(outs_dir / "respondent_dev.csv.gz")
    dev         = dev_df["dev"].to_numpy(dtype=float)
    country_idx = dev_df["country_idx"].to_numpy(dtype=int)

    # ── Weights ──────────────────────────────────────────────────────────────
    wdf      = pd.read_csv(output_dir / "weights.csv.gz")
    pspwght  = wdf["pspwght"].to_numpy(dtype=float)
    weights_by: dict[str, np.ndarray] = {
        s: wdf[f"final_weight_{s}"].to_numpy(dtype=float) for s in SETTING_NAMES
    }

    # ── Country-level point estimates ────────────────────────────────────────
    cm_pe    = pd.read_csv(outs_dir / "country_means_point_estimates.csv")
    op_by_setting = {name: op for name, op, _ in SETTING_DEFS}

    # ── Bootstrap CIs (optional) ─────────────────────────────────────────────
    boot_ci_path = boot_dir / "country_mean_ci.csv"
    country_mean_ci_df = pd.read_csv(boot_ci_path) if boot_ci_path.exists() else None

    # ── Propensity histograms ─────────────────────────────────────────────────
    prop_df = pd.read_csv(output_dir / "propensity_oof.csv.gz")
    proba   = prop_df[[c for c in prop_df.columns if c.startswith("p_")]].to_numpy(dtype=float)
    print("  [diag] Propensity histograms …")
    plot_propensity_histograms(
        proba, country_idx, classes,
        diag_dir / "propensity_histograms.png", qset_key,
    )

    # ── Balance plots ─────────────────────────────────────────────────────────
    bal_var     = pd.read_csv(diag_dir / "balance_per_variable.csv")
    bal_country = pd.read_csv(diag_dir / "balance_per_country.csv")
    print("  [diag] Balance plots …")
    plot_balance_per_variable(bal_var, diag_dir / "balance_per_variable.png", qset_key)
    plot_balance_per_country(bal_country, diag_dir / "balance_per_country.png", qset_key)

    # ── Country-means three-stage forest (per setting) ────────────────────────
    print("  [outputs] Country-means three-stage plots …")
    for setting in SETTING_NAMES:
        fw      = weights_by[setting]
        op      = op_by_setting[setting]
        suffix  = "" if setting == PRIMARY_SETTING else f"_{setting}"

        sub_rew = cm_pe[(cm_pe["setting"] == setting) & (cm_pe["series"] == "reweighted")]
        df_country_rew = sub_rew[["country", "mean"]].reset_index(drop=True)

        sub_surv = cm_pe[(cm_pe["setting"] == setting) & (cm_pe["series"] == "pspwght_survivors")]
        pspwght_survivors_means = (
            sub_surv[["country", "mean"]].reset_index(drop=True)
            if op == "drop" and not sub_surv.empty else None
        )

        cm_ci_rew    = None
        cm_ci_pwsurv = None
        if country_mean_ci_df is not None and "setting" in country_mean_ci_df.columns:
            sub_ci = country_mean_ci_df[country_mean_ci_df["setting"] == setting]
            if "series" in sub_ci.columns:
                cm_ci_rew_sub = sub_ci[sub_ci["series"] == "reweighted"].drop(
                    columns=["setting", "series"]).reset_index(drop=True)
                cm_ci_rew = cm_ci_rew_sub if not cm_ci_rew_sub.empty else None
                cm_ci_surv_sub = sub_ci[sub_ci["series"] == "pspwght_survivors"].drop(
                    columns=["setting", "series"]).reset_index(drop=True)
                cm_ci_pwsurv = cm_ci_surv_sub if not cm_ci_surv_sub.empty else None
            else:
                cm_ci_rew = sub_ci.drop(columns=["setting"]).reset_index(drop=True)

        plot_country_means_three_stage(
            df_country=df_country_rew,
            country_mean_ci=cm_ci_rew,
            df=pd.DataFrame(),  # not used in plot logic
            country_idx=country_idx,
            classes=classes,
            dev=dev,
            pspwght=pspwght,
            fw_primary=fw,
            output_path=outs_dir / f"country_means_three_stage{suffix}.png",
            question_set_key=qset_key,
            pspwght_survivors_means=pspwght_survivors_means,
            pspwght_survivors_ci=cm_ci_pwsurv,
        )

    # ── Before-after deviation plot ───────────────────────────────────────────
    print("  [outputs] Cross-model deviation before-after plot …")
    llm_dev_df = pd.read_csv(outs_dir / "per_llm_country_deviations.csv")

    def _rebuild_llm_dev(stage: str) -> dict[str, dict[str, float]]:
        sub = llm_dev_df[llm_dev_df["stage"] == stage]
        out: dict[str, dict[str, float]] = {}
        for row in sub.itertuples(index=False):
            out.setdefault(row.country, {})[row.llm_model] = float(row.deviation)
        return out

    per_llm_dev_pre  = _rebuild_llm_dev("pre")
    per_llm_dev_post = _rebuild_llm_dev("post")
    cm_pre_df = pd.DataFrame({
        "country": classes,
        "mean": [float(np.mean(list(per_llm_dev_pre.get(c, {}).values()) or [np.nan]))
                 for c in classes],
    })
    sub_post = cm_pe[(cm_pe["setting"] == PRIMARY_SETTING) & (cm_pe["series"] == "reweighted")]
    cm_post_df = sub_post[["country", "mean"]].reset_index(drop=True)
    plot_cross_model_deviation_before_after(
        classes=classes,
        per_country_cross_model_dev_pre=cm_pre_df,
        per_country_cross_model_dev_post=cm_post_df,
        per_llm_country_dev_pre=per_llm_dev_pre,
        per_llm_country_dev_post=per_llm_dev_post,
        output_path=outs_dir / "cross_model_deviation_before_after_clip_p01.png",
        question_set_key=qset_key,
    )

    print(f"  IPW plots regenerated in {output_dir}")


def run_ipw(
    db_path: Path,
    question_set_key: str,
    output_dir: Path,
    model_registry: dict = MODEL_REGISTRY,
    weight_col: str = "pspwght",
    n_bootstrap: int = BOOTSTRAP_REPLICATES_DEFAULT,
    crossfit_folds: int = CROSSFIT_FOLDS,
    seed: int = RANDOM_SEED,
    n_jobs: int = -1,
    propensity_params: dict | None = None,
    skip_bootstrap: bool = False,
) -> None:
    """
    IPW analysis for one question set.

    Loads the qset, fits one cross-fitted propensity, evaluates the three
    settings (clip_p01 + drop_p01 + drop_p05), writes point-estimate
    artefacts, and optionally bootstraps for CIs.

    Each bootstrap replicate emits two CI channels: per-country cross-model
    means (with a ``pspwght_survivors`` series for drop settings) and per-LLM
    between-country σ_pre / σ_post — the latter feeds the variance tables'
    95% CI on Δσ.
    """
    print(f"\n  Loading IPW dataset for qset={question_set_key!r} …")
    qset_df, model_shorts = load_ipw_dataset(
        db_path, question_set_key, model_registry, weight_col,
    )
    qset_df = qset_df.reset_index(drop=True)
    print(f"    {question_set_key}: {len(qset_df):,} respondents, "
          f"{len(model_shorts)} models")

    # Outer propensity fit gets all cores; per-replicate fits inside the
    # bootstrap default to single-thread so joblib's process-level
    # parallelism doesn't oversubscribe.
    print(f"\n  [1/3] Fitting cross-fitted propensity …")
    t0 = time.time()
    qset_oof, classes, run_meta = fit_crossfitted_propensity(
        qset_df,
        n_splits=crossfit_folds,
        params=propensity_params,
        seed=seed,
        n_jobs=-1,
    )
    print(f"        Done in {(time.time()-t0)/60:.1f} min")

    print(f"\n  [2/3] Point-estimate evaluation …")
    eval_res = _evaluate_qset(
        qset_df, qset_oof, model_shorts, classes, seed=seed,
    )

    sd_present = [v for v in IPW_SD_COVARIATES if v in qset_df.columns]

    print(f"\n  [3/4] Writing point-estimate artifacts …")
    _write_qset_artifacts(
        output_dir=output_dir,
        qset_key=question_set_key,
        eval_res=eval_res,
        model_shorts=model_shorts,
        classes=classes,
        weight_col=weight_col,
        seed=seed,
        union_meta=run_meta,
        sd_present=sd_present,
        n_bootstrap=0,
        crossfit_folds=crossfit_folds,
        df_boot_res=None,
        df_boot_cm=None,
    )

    if skip_bootstrap or n_bootstrap <= 0:
        print(f"\n  [4/4] Bootstrap skipped")
        print(f"\n  IPW analysis complete for {question_set_key!r}.")
        return

    print(f"\n  [4/4] Bootstrap (B={n_bootstrap}, K={crossfit_folds}) …")
    df_boot_res, df_boot_cm, df_boot_sigma = run_bootstrap(
        qset_df, model_shorts,
        n_replicates=n_bootstrap,
        crossfit_folds=crossfit_folds,
        propensity_params=propensity_params or {},
        base_seed=seed,
        n_jobs=n_jobs,
    )

    print(f"\n  Augmenting artifacts with bootstrap CIs …")
    _write_qset_artifacts(
        output_dir=output_dir,
        qset_key=question_set_key,
        eval_res=eval_res,
        model_shorts=model_shorts,
        classes=classes,
        weight_col=weight_col,
        seed=seed,
        union_meta=run_meta,
        sd_present=sd_present,
        n_bootstrap=int(n_bootstrap),
        crossfit_folds=crossfit_folds,
        df_boot_res=df_boot_res,
        df_boot_cm=df_boot_cm,
        df_boot_sigma=df_boot_sigma,
    )

    print(f"\n  IPW analysis complete for {question_set_key!r}.")


