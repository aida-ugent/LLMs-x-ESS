"""
DB I/O and data loading utilities.

Functions
---------
load_questions        Filter value_laden_Qs.csv to a question set.
list_question_sets    Enumerate all valid --question-set values.
load_ess_data         Load merged ess11 + demographics from DB.
load_majority_votes   Load and concatenate model majority-vote CSVs.
write_alignments_to_db     Write person-level alignment scores into demographics.
write_per_question_to_db   Write per-question scores into a new DB table.
"""

import sqlite3
import warnings
from pathlib import Path

import pandas as pd

from .config import DEFAULT_QUESTIONS_CSV, DEFAULT_MAJORITY_VOTES_CSVS, MODEL_REGISTRY


# ── Question-set helpers ───────────────────────────────────────────────────────

def q_set_key(question_set: str) -> str:
    """Return a filesystem- and SQLite-column-safe key for *question_set*."""
    return question_set.strip().lower().replace(" ", "_").replace("/", "_")


def load_questions(
    question_set: str,
    questions_csv=DEFAULT_QUESTIONS_CSV,
    majority_votes_csvs=None,
    model_registry: dict = MODEL_REGISTRY,
) -> pd.DataFrame:
    """
    Filter question metadata to the requested question set.

    Parameters
    ----------
    question_set : str
        ``"all"``           – every question with has_question == True.
        ``"just_values"``   – questions where value_survey == 1.
        ``"non_value"``     – questions where value_survey != 1.
        ``"all_answered"``  – questions answered by ALL models in MODEL_REGISTRY
                              (intersection across majority-vote CSVs).
        Any *topic* string (case-insensitive) – questions for that topic.

    Returns
    -------
    pd.DataFrame  with columns from value_laden_Qs.csv.
    """
    df = pd.read_csv(questions_csv)

    # Normalise has_question (may arrive as bool or string "TRUE"/"FALSE")
    df["has_question"] = df["has_question"].astype(str).str.upper().isin(
        ["TRUE", "1", "YES"]
    )
    df = df[df["has_question"]].copy()

    if question_set == "all":
        pass
    elif question_set == "just_values":
        df = df[pd.to_numeric(df["value_survey"], errors="coerce") == 1]
    elif question_set == "non_value":
        df = df[pd.to_numeric(df["value_survey"], errors="coerce") != 1]
    elif question_set == "all_answered":
        csvs = majority_votes_csvs or DEFAULT_MAJORITY_VOTES_CSVS
        dfs_votes = []
        for p in csvs:
            p = Path(p)
            if p.exists():
                dfs_votes.append(pd.read_csv(p))
        if not dfs_votes:
            raise FileNotFoundError(
                "'all_answered' requires majority-vote CSVs but none were found. "
                "Run Stage 0 first or pass --votes-csv explicitly."
            )
        votes = pd.concat(dfs_votes, ignore_index=True)
        # Keep only models present in the registry
        votes = votes[votes["model"].isin(model_registry)]
        registered_models = set(votes["model"].unique())
        if not registered_models:
            raise ValueError(
                "No models in MODEL_REGISTRY found in the majority-vote CSVs."
            )
        # Intersection: questions with a non-NaN answer from every registered model
        valid_votes = votes.dropna(subset=["majority_vote_answer"])
        questions_per_model = (
            valid_votes.groupby("model")["question_var"]
            .apply(set)
        )
        common_questions = set.intersection(*questions_per_model.tolist())
        df = df[df["var_name"].isin(common_questions)]
        print(
            f"    all_answered: {len(registered_models)} models, "
            f"{len(common_questions)} common questions "
            f"(intersection of: {sorted(registered_models)})"
        )
    else:
        topics = df["topic"].str.lower().str.strip().unique()
        qs_lower = question_set.lower().strip()
        if qs_lower not in topics:
            available = sorted(
                {"all", "all_answered", "just_values", "non_value"} | set(topics)
            )
            raise ValueError(
                f"Unknown question_set '{question_set}'.\n"
                f"Available: {available}\n"
                f"Use --list-question-sets to display them."
            )
        df = df[df["topic"].str.lower().str.strip() == qs_lower]

    print(f"  Question set '{question_set}': {len(df)} questions")
    if len(df) == 0:
        raise ValueError(f"Question set '{question_set}' produced 0 questions.")
    return df.reset_index(drop=True)


def list_question_sets(questions_csv=DEFAULT_QUESTIONS_CSV) -> list[str]:
    """Return sorted list of all valid --question-set values."""
    df = pd.read_csv(questions_csv)
    topics = sorted(df["topic"].str.lower().str.strip().dropna().unique().tolist())
    return ["all", "all_answered", "just_values", "non_value"] + topics


# ── ESS data loading ───────────────────────────────────────────────────────────

def load_ess_data(db_path) -> pd.DataFrame:
    """
    Load merged ess11 + demographics from the ESS11 SQLite database.

    Returns a single DataFrame joined on new_id, with demographics columns
    appended (avoiding duplicate columns).
    """
    conn = sqlite3.connect(db_path)
    df_ess = pd.read_sql_query("SELECT * FROM ess11", conn)
    df_demo = pd.read_sql_query("SELECT * FROM demographics", conn)
    conn.close()

    # Only bring in demographics columns that are not already in ess11
    extra_demo_cols = ["new_id"] + [
        c for c in df_demo.columns if c not in df_ess.columns
    ]
    df = df_ess.merge(df_demo[extra_demo_cols], on="new_id", how="left")

    print(f"  ESS data loaded: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ── Majority-vote loading ──────────────────────────────────────────────────────

def load_majority_votes(
    csv_paths=None,
    model_registry: dict = MODEL_REGISTRY,
) -> pd.DataFrame:
    """
    Load and concatenate model majority-vote CSVs.

    Only models present in *model_registry* are kept; others produce a warning.
    """
    if csv_paths is None:
        csv_paths = DEFAULT_MAJORITY_VOTES_CSVS

    dfs = []
    for p in csv_paths:
        p = Path(p)
        if not p.exists():
            warnings.warn(f"Majority votes CSV not found: {p}")
            continue
        df = pd.read_csv(p)
        dfs.append(df)
        print(
            f"  Loaded votes: {p.name} "
            f"({df['model'].nunique()} models, {df['question_var'].nunique()} questions)"
        )

    if not dfs:
        raise FileNotFoundError("No majority-vote CSVs could be loaded.")

    combined = pd.concat(dfs, ignore_index=True)

    unknown = set(combined["model"].unique()) - set(model_registry)
    if unknown:
        warnings.warn(
            f"Models not in MODEL_REGISTRY (skipped): {unknown}\n"
            "Add them to pipeline/config.py → MODEL_REGISTRY to include them."
        )
    combined = combined[combined["model"].isin(model_registry)]

    print(
        f"  Combined votes: {combined['model'].nunique()} models, "
        f"{combined['question_var'].nunique()} questions"
    )
    return combined


# ── DB write helpers ───────────────────────────────────────────────────────────

def write_alignments_to_db(
    db_path,
    df_person: pd.DataFrame,
    question_set_key: str,
    model_registry: dict = MODEL_REGISTRY,
) -> None:
    """
    Add/update alignment columns in the demographics table.

    Columns added for each model:
      {model_short}_{question_set_key}_alignment   REAL
      n_q_{model_short}_{question_set_key}          INTEGER

    Existing columns are kept; only rows with a non-null value are updated.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Index on new_id makes each UPDATE…WHERE new_id=? an O(log n) lookup
    # instead of a full table scan — ~3000x faster for 50k rows × 10 columns.
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_demographics_new_id "
        "ON demographics (new_id)"
    )
    conn.commit()

    for model_full, model_short in model_registry.items():
        src_align = f"{model_short}_alignment"
        src_nq    = f"n_q_{model_short}"
        dst_align = f"{model_short}_{question_set_key}_alignment"
        dst_nq    = f"n_q_{model_short}_{question_set_key}"

        if src_align not in df_person.columns:
            continue  # this model had no votes for this question set

        # Ensure columns exist
        for col, ctype in [(dst_align, "REAL"), (dst_nq, "INTEGER")]:
            try:
                cursor.execute(
                    f'ALTER TABLE demographics ADD COLUMN "{col}" {ctype}'
                )
            except sqlite3.OperationalError:
                pass  # column already exists

        # Update alignment
        rows = (
            df_person[["new_id", src_align]]
            .dropna(subset=[src_align])
            .itertuples(index=False, name=None)
        )
        cursor.executemany(
            f'UPDATE demographics SET "{dst_align}" = ? WHERE new_id = ?',
            [(float(a), nid) for nid, a in rows],
        )

        # Update n_q
        if src_nq in df_person.columns:
            rows_nq = (
                df_person[["new_id", src_nq]]
                .dropna(subset=[src_nq])
                .itertuples(index=False, name=None)
            )
            cursor.executemany(
                f'UPDATE demographics SET "{dst_nq}" = ? WHERE new_id = ?',
                [(int(n), nid) for nid, n in rows_nq],
            )

    conn.commit()
    conn.close()
    print(
        f"  demographics table updated with alignments "
        f"for question set '{question_set_key}'"
    )


def write_per_question_to_db(
    db_path,
    df_wide: pd.DataFrame,
    question_set_key: str,
) -> None:
    """
    Create (or replace) the per_question_{question_set_key} table.

    Schema
    ------
    new_id TEXT PRIMARY KEY
    {model_short}_{question_var} REAL   – NULL if person did not answer
    ...
    """
    table_name = f"per_question_{question_set_key}"
    conn = sqlite3.connect(db_path)
    df_wide.to_sql(table_name, conn, if_exists="replace", index=False)
    # Index so downstream JOIN/WHERE on new_id is fast.
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_new_id "
        f'ON "{table_name}" (new_id)'
    )
    conn.commit()
    conn.close()
    print(
        f"  Created table '{table_name}': "
        f"{df_wide.shape[0]:,} rows × {df_wide.shape[1]} columns"
    )
