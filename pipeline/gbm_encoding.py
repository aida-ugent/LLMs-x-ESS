"""
Feature encoding helpers for native categorical handling in gradient-boosted
tree models. Used by `method_comp.py` GBM rows; LR rows use the one-hot
encoders defined there.

Two encodings exposed via `prepare_features`:
  - "onehot": dense float numpy array, identical to the matrix built by
              `method_comp._sd_dummies` / `_country_sd_dummies`.
  - "native": pandas DataFrame with ordinal SD vars cast to numeric and
              nominal SD vars cast to pandas Categorical, ready for any
              backend that supports `enable_categorical=True` (XGBoost),
              `categorical_feature=` (LightGBM), or `cat_features=` (CatBoost).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# SD variable typing — nominal vs ordinal
# ─────────────────────────────────────────────────────────────────────────────

# Truly nominal vars (and country): no ordering, dtype = pd.Categorical.
NOMINAL_VARS = {
    "gender_clean",
    "ethnicmajority_clean",
    "religion_clean",
    "mnactic_clean",
    "isco08_major_group",
    "immigration_background",
    "cntry",
}

# Numeric-but-ordinal SD vars: stored as floats, preserve order as-is.
ORDINAL_NUMERIC_VARS = {
    "income_clean",             # 1-10
    "political_interest_clean", # 1-4
    "political_involvement",    # 0-7 count
    "domicil_clean",            # 1-5
    "hincfel_clean",            # 1-4
    "fnsdfml_clean",            # 1-5
}

# String-but-ordinal SD vars: map to integer rank via explicit order.
ORDINAL_STRING_ORDERS = {
    "education_bracket": [
        "0-1: ≤ Lower secondary",
        "2: Lower secondary",
        "3: Upper secondary",
        "4: Post-secondary non-tertiary",
        "5: Short-cycle tertiary",
        "6: Bachelor",
        "7: Master",
        "8: Doctoral",
    ],
    "religiosity_clean": [
        "0-2: Low religiosity",
        "3-7: Medium religiosity",
        "8-10: High religiosity",
    ],
    "internet_time_bracket": [
        "0: No use",
        "1-30: Up to 1/2 hour",
        "31-60: 1/2-1 hour",
        "61-180: 1-3 hours",
        "181-300: 3-5 hours",
        "301-420: 5-7 hours",
        "421+: More than 7 hours",
    ],
    "generations": [
        "Gen Z",
        "Millennials",
        "Gen X",
        "Baby Boomers",
        "Silent Generation",
    ],
}


@dataclass
class EncodedFeatures:
    """Container returned by `prepare_features`.

    onehot : X is a dense float numpy array; categorical_cols is empty.
    native : X is a pandas DataFrame (mixed numeric + Categorical dtypes);
             categorical_cols lists the Categorical column names.
    """
    X: object                  # np.ndarray (onehot) or pd.DataFrame (native)
    feature_names: list[str]
    categorical_cols: list[str]
    encoding: str              # "onehot" | "native"


def _onehot_matrix(df: pd.DataFrame, cols: list[str], with_country: bool) -> tuple[np.ndarray, list[str]]:
    """Mirror of method_comp._sd_dummies / _country_sd_dummies, kept here so
    other callers can request the same matrix without going through method_comp.

    Uses the variable name as the dummy prefix (pandas default) so identical
    category values across variables (e.g. '1.0' for both gender and income)
    do not collide into the same column label.
    """
    cat_cols = [c for c in cols if c in df.columns]
    df_enc = df[cat_cols].astype(str).replace("nan", "__missing__")
    sd = pd.get_dummies(df_enc)
    sd = sd[sorted(sd.columns.tolist())]
    if with_country:
        ctry = pd.get_dummies(df["cntry"].astype(str), prefix="cntry")
        out = pd.concat([ctry, sd], axis=1)
    else:
        out = sd
    return out.values.astype(float), out.columns.tolist()


def _native_frame(df: pd.DataFrame, cols: list[str], with_country: bool) -> tuple[pd.DataFrame, list[str]]:
    """Build a mixed-dtype DataFrame: ordinal vars numeric, nominal vars Categorical.

    Returns (X_df, categorical_cols).
    """
    out_cols: dict[str, pd.Series] = {}
    categorical_cols: list[str] = []
    present = [c for c in cols if c in df.columns]

    for c in present:
        s = df[c]
        if c in ORDINAL_NUMERIC_VARS:
            out_cols[c] = pd.to_numeric(s, errors="coerce")
        elif c in ORDINAL_STRING_ORDERS:
            order = ORDINAL_STRING_ORDERS[c]
            idx_map = {v: i for i, v in enumerate(order)}
            out_cols[c] = s.map(idx_map).astype(float)  # NaN preserved
        else:  # nominal
            sv = s.astype(object).where(s.notna(), "__missing__").astype(str)
            cats = sorted(sv.unique().tolist())
            out_cols[c] = pd.Categorical(sv, categories=cats)
            categorical_cols.append(c)

    if with_country:
        sv = df["cntry"].astype(object).where(df["cntry"].notna(), "__missing__").astype(str)
        cats = sorted(sv.unique().tolist())
        out_cols["cntry"] = pd.Categorical(sv, categories=cats)
        categorical_cols.append("cntry")

    X_df = pd.DataFrame(out_cols, index=df.index)
    return X_df, categorical_cols


def prepare_features(
    df: pd.DataFrame,
    cols: list[str],
    *,
    with_country: bool,
    encoding: str,
) -> EncodedFeatures:
    """Build a feature matrix in the requested encoding.

    `cols` is the list of SD variable names to include (typically
    `method_comp.COVARIATE_VARS`; missing columns are skipped silently).
    """
    if encoding == "onehot":
        X, names = _onehot_matrix(df, cols, with_country)
        return EncodedFeatures(X=X, feature_names=names, categorical_cols=[], encoding="onehot")
    if encoding == "native":
        X_df, cat_cols = _native_frame(df, cols, with_country)
        return EncodedFeatures(
            X=X_df, feature_names=list(X_df.columns),
            categorical_cols=cat_cols, encoding="native",
        )
    raise ValueError(f"Unknown encoding: {encoding!r}")


def slice_features(enc: EncodedFeatures, idx: np.ndarray) -> EncodedFeatures:
    """Row-slice a feature matrix preserving its container type and metadata."""
    if enc.encoding == "onehot":
        X = enc.X[idx]
    else:
        X = enc.X.iloc[idx].reset_index(drop=True)
    return EncodedFeatures(
        X=X,
        feature_names=enc.feature_names,
        categorical_cols=enc.categorical_cols,
        encoding=enc.encoding,
    )
