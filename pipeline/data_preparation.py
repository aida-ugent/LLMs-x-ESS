"""
Build the `ess11` and `demographics` tables in ESS11.db from the official
ESS11e04_0 export.

Main entry points
-----------------
build_ess11_db(raw_csv_path, db_path)
    Reads ESS11e04_0.csv, derives a stable `new_id` (idno_cntry_essround_yrbrn),
    and (re)writes the `ess11` table in *db_path*.

build_demographics_table(db_path, codebook_html_path)
    Rebuilds the `demographics` table from `ess11` plus the value labels in the
    official codebook HTML. Must run *after* build_ess11_db.
"""

from __future__ import annotations

import html
import re
import sqlite3
from pathlib import Path

import pandas as pd

ID_COLUMNS = ["idno", "cntry", "essround", "yrbrn"]

# ── demographics layout ───────────────────────────────────────────────────────
# Raw ESS columns copied verbatim from `ess11`, in table order. These duplicate
# `ess11`, so `load_ess_data()` drops them again on merge; they are kept so the
# table stands on its own as a socio-demographic extract.
RAW_COLUMNS: list[str] = [
    "new_id", "idno", "nwspol", "netusoft", "netustm", "polintr", "vote",
    "contplt", "donprty", "badge", "sgnptit", "pbldmna", "bctprd", "pstplonl",
    "volunfp", "stflife", "happy", "sclmeet", "inprdsc", "sclact", "atchctr",
    "atcherp", "rlgblg", "rlgdnm", "rlgblge", "rlgdnme", "rlgdgr", "rlgatnd",
    "pray", "dscrgrp", "ctzcntr", "brncntr", "cntbrthd", "livecnta",
    "feethngr", "facntr", "fbrncntc", "mocntr", "mbrncntc", "nobingnd",
    "likrisk", "liklead", "sothnds", "actcomp", "hhmmb", "chldhhe", "domicil",
    "edulvlb", "eduyrs", "mainact", "crpdwk", "pdjobev", "pdjobyr", "emplrel",
    "wrkac6m", "uemp3m", "mbtru", "hincsrca", "hinctnta", "hincfel",
    "edulvlpb", "crpdwkp", "emprelp", "wkhtotp", "edulvlfb", "emprf14",
    "occf14b", "edulvlmb", "emprm14", "occm14b",
]

# Variables that additionally get a `{var}_labeled` / `{var}_clean` pair, in
# table order (which is not the RAW_COLUMNS order).
LABELLED_VARIABLES: list[str] = [
    "netusoft", "vote", "contplt", "donprty", "badge", "sgnptit", "pbldmna",
    "bctprd", "pstplonl", "volunfp", "stflife", "happy", "sclmeet", "inprdsc",
    "sclact", "rlgblg", "rlgdnm", "rlgblge", "rlgdnme", "rlgatnd", "pray",
    "dscrgrp", "ctzcntr", "brncntr", "cntbrthd", "feethngr", "facntr",
    "fbrncntc", "mocntr", "mbrncntc", "nobingnd", "chldhhe", "domicil",
    "edulvlb", "mainact", "crpdwk", "pdjobev", "emplrel", "wrkac6m", "uemp3m",
    "mbtru", "hincsrca", "hinctnta", "hincfel", "edulvlpb", "crpdwkp",
    "emprelp", "edulvlfb", "emprf14", "occf14b", "edulvlmb", "emprm14",
    "occm14b",
]

# Country-of-birth variables. Their codebook lists ISO-3166 alpha-2 country
# codes alongside UN region aggregates (2 = Africa, 150 = Europe, …) and
# historical states (1000 = DDR, 2000 = USSR, …). Only a real country counts as
# an answer here, so anything that is not alpha-2 is treated as non-informative:
# `_labeled` keeps the raw code and `_clean` is NULL.
ISO_COUNTRY_VARIABLES: frozenset[str] = frozenset(
    {"cntbrthd", "fbrncntc", "mbrncntc"}
)

_ISO_ALPHA2 = re.compile(r"^[A-Z]{2}$")

# <h3 id="varname"> … <tbody class="codelist"> <tr><td class="nowrap">code</td>
# <td>label</td></tr> … </tbody>
_VAR_BLOCK_RE = re.compile(r'<h3 id="([^"]+)">')
_CODELIST_RE = re.compile(r'<tbody class="codelist">(.*?)</tbody>', re.S)
_CODE_ROW_RE = re.compile(
    r'<td class="nowrap">(.*?)</td>\s*<td>(.*?)</td>', re.S
)


def build_ess11_db(raw_csv_path: Path, db_path: Path) -> pd.DataFrame:
    """
    Build the `ess11` table from the raw ESS11e04_0.csv export.

    Parameters
    ----------
    raw_csv_path : Path to ESS11e04_0.csv (official ESS Round 11 data file).
    db_path      : Path to ESS11.db. Created if it doesn't exist yet; if it
                   does, only the `ess11` table is replaced.

    Returns
    -------
    pd.DataFrame  The DataFrame written to the `ess11` table.
    """
    raw_csv_path = Path(raw_csv_path)
    db_path = Path(db_path)

    print("\n" + "=" * 60)
    print("STAGE – Data preparation (build ess11 table)")
    print("=" * 60)

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw ESS11 CSV not found: {raw_csv_path}")

    print(f"\n  Reading: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    missing = [c for c in ID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV is missing expected id columns: {missing}")

    # new_id must match the convention used elsewhere in the pipeline
    # (demographics table, majority-vote alignment, etc.).
    new_id = (
        df["idno"].astype(str) + "_" +
        df["cntry"].astype(str) + "_" +
        df["essround"].astype(str) + "_" +
        df["yrbrn"].astype(str)
    )
    df = pd.concat([new_id.rename("new_id"), df], axis=1)


    n_dup = df["new_id"].duplicated().sum()
    if n_dup:
        print(f"  WARNING: {n_dup} duplicate new_id values found")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("ess11", conn, if_exists="replace", index=False)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_ess11_new_id ON ess11 (new_id)"
    )
    conn.commit()
    conn.close()

    print(
        f"  Wrote table 'ess11': {df.shape[0]:,} rows × {df.shape[1]} columns "
        f"→ {db_path}"
    )
    return df


# ── Codebook value labels ─────────────────────────────────────────────────────

def parse_codebook_labels(
    codebook_html_path: Path,
) -> dict[str, dict[str, tuple[str, bool]]]:
    """
    Extract the value labels for every variable in the official codebook HTML.

    The codebook renders one ``<h3 id="varname">`` block per variable, each with
    a ``<tbody class="codelist">`` table of value/category rows. Missing-value
    categories are marked with a trailing asterisk (``Refusal*``, ``Don't know*``,
    ``No answer*``, ``Not applicable*``), which is what lets the recode below
    decide per code whether it counts as an answer — no hand-maintained list of
    missing codes is needed.

    Returns
    -------
    dict  ``{var_name: {code_as_str: (label_without_asterisk, is_missing)}}``
    """
    codebook_html_path = Path(codebook_html_path)
    if not codebook_html_path.exists():
        raise FileNotFoundError(
            f"ESS11 codebook HTML not found: {codebook_html_path}"
        )

    text = codebook_html_path.read_text(encoding="utf-8", errors="replace")

    labels: dict[str, dict[str, tuple[str, bool]]] = {}
    blocks = _VAR_BLOCK_RE.split(text)
    # blocks[0] is everything before the first <h3 id=…>, then (var, body) pairs
    for i in range(1, len(blocks) - 1, 2):
        var, body = blocks[i], blocks[i + 1]
        codelist = _CODELIST_RE.search(body)
        if not codelist:
            continue
        codes: dict[str, tuple[str, bool]] = {}
        for code, label in _CODE_ROW_RE.findall(codelist.group(1)):
            code = html.unescape(code.strip())
            label = html.unescape(label.strip())
            is_missing = label.endswith("*")
            codes[code] = (label[:-1] if is_missing else label, is_missing)
        if codes:
            labels[var] = codes

    print(f"  Codebook: value labels for {len(labels)} variables")
    return labels


def _as_code(value) -> str | None:
    """
    Render a raw ESS value as the string used to look codes up in the codebook.

    Whole-number floats must render as ``"7"`` rather than ``"7.0"``, otherwise
    every lookup misses.
    """
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    return str(value).strip()


def _recode_variable(
    values: pd.Series,
    code_labels: dict[str, tuple[str, bool]],
    iso_country_only: bool,
) -> tuple[list[str | None], list[str | None]]:
    """
    Map one raw ESS column to its ``_labeled`` / ``_clean`` pair.

    ==========================================  ============  ==========
    case                                        ``_labeled``  ``_clean``
    ==========================================  ============  ==========
    raw value missing                           NULL          NULL
    code carries a real category label          the label     the label
    code carries a missing-value label (``*``)  ``str(code)`` NULL
    code has no codebook entry                  ``str(code)`` ``str(code)``
    ==========================================  ============  ==========

    For the country-of-birth variables, a code that is not an ISO-3166 alpha-2
    country is treated as non-informative regardless of its codebook label.
    """
    labeled: list[str | None] = []
    clean: list[str | None] = []

    for raw in values:
        code = _as_code(raw)
        if code is None:
            labeled.append(None)
            clean.append(None)
        elif iso_country_only and not _ISO_ALPHA2.match(code):
            labeled.append(code)
            clean.append(None)
        elif code in code_labels:
            label, is_missing = code_labels[code]
            labeled.append(code if is_missing else label)
            clean.append(None if is_missing else label)
        else:
            labeled.append(code)
            clean.append(code)

    return labeled, clean


def build_demographics_table(
    db_path: Path,
    codebook_html_path: Path,
) -> pd.DataFrame:
    """
    Rebuild the `demographics` table from `ess11` + the codebook value labels.

    The table holds the socio-demographic subset of the ESS variables, each in
    three forms: the raw code, a ``{var}_labeled`` column carrying the codebook
    category text, and a ``{var}_clean`` column that is NULL wherever the answer
    is a missing-value code. The pipeline reads ``domicil_clean`` from here
    (see ``pipeline/bootstrap.py``); the remaining columns document the sample.

    Alignment columns (``{model}_{qset}_alignment`` / ``n_q_{model}_{qset}``) are
    *not* created here — ``pipeline/data.write_alignments_to_db`` adds them on
    demand with ALTER TABLE. Rebuilding this table therefore discards previously
    computed alignments, which is intended: they are re-derived by Stage 1.

    Parameters
    ----------
    db_path            : Path to ESS11.db, which must already contain `ess11`.
    codebook_html_path : Path to "ESS11e04_0 codebook.html".

    Returns
    -------
    pd.DataFrame  The DataFrame written to the `demographics` table.
    """
    db_path = Path(db_path)

    print("\n" + "=" * 60)
    print("STAGE – Data preparation (build demographics table)")
    print("=" * 60)

    code_labels = parse_codebook_labels(codebook_html_path)

    conn = sqlite3.connect(db_path)
    ess_cols = {r[1] for r in conn.execute("PRAGMA table_info(ess11)")}
    if not ess_cols:
        conn.close()
        raise RuntimeError(
            f"No `ess11` table in {db_path}. Run the ess11 build first "
            "(python run_pipeline.py --data-preparation)."
        )

    missing = [c for c in RAW_COLUMNS if c not in ess_cols]
    if missing:
        conn.close()
        raise ValueError(
            f"`ess11` is missing expected socio-demographic columns: {missing}"
        )

    quoted = ", ".join(f'"{c}"' for c in RAW_COLUMNS)
    df = pd.read_sql_query(f"SELECT {quoted} FROM ess11", conn)
    print(f"  Read {len(df):,} rows from `ess11` ({len(RAW_COLUMNS)} raw columns)")

    # Build the labeled/clean pairs in table order.
    recoded: dict[str, list[str | None]] = {}
    no_labels: list[str] = []
    for var in LABELLED_VARIABLES:
        var_labels = code_labels.get(var, {})
        if not var_labels:
            no_labels.append(var)
        labeled, clean = _recode_variable(
            df[var], var_labels, var in ISO_COUNTRY_VARIABLES
        )
        recoded[f"{var}_labeled"] = labeled
        recoded[f"{var}_clean"] = clean

    if no_labels:
        print(
            f"  WARNING: no codebook labels found for {len(no_labels)} "
            f"variable(s): {no_labels}"
        )

    df = pd.concat([df, pd.DataFrame(recoded, index=df.index)], axis=1)
    print(f"  Recoded {len(LABELLED_VARIABLES)} variables → labeled/clean pairs")

    df.to_sql("demographics", conn, if_exists="replace", index=False)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_demographics_new_id "
        "ON demographics (new_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_demographics_idno "
        "ON demographics (idno)"
    )
    conn.commit()
    conn.close()

    print(
        f"  Wrote table 'demographics': {df.shape[0]:,} rows × "
        f"{df.shape[1]} columns → {db_path}"
    )
    return df
