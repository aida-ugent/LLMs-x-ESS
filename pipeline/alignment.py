"""
Alignment score calculation

calculate_alignments(df_ess, df_votes, questions_df, ...) → (df_person, df_pq)

Formula
-----------------------
Per-question score:
    a_{p,q,m} = 1 - |answer_p_q - vote_m_q| / (max_scale_q - min_scale_q)
    NaN if person did not give a valid answer for that question.

Person-level score:
    1. Questions that belong to the same QUESTION_GROUP_MAP group are first
       averaged within the group (they count as one question).
    2. The person-level alignment is the mean across all question groups for
       which at least one sub-question was answered.
"""

import warnings

import numpy as np
import pandas as pd

from .config import MODEL_REGISTRY, QUESTION_GROUP_MAP


def calculate_alignments(
    df_ess: pd.DataFrame,
    df_votes: pd.DataFrame,
    questions_df: pd.DataFrame,
    question_group_map: dict = QUESTION_GROUP_MAP,
    model_registry: dict = MODEL_REGISTRY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorised alignment calculation for all person × model pairs.

    Parameters
    ----------
    df_ess        Merged ESS11 + demographics DataFrame (output of load_ess_data).
    df_votes      Majority-vote DataFrame (output of load_majority_votes).
    questions_df  Question metadata filtered to the current question set.
    question_group_map  Maps question_var → group key for sub-question averaging.
    model_registry      Maps full model name → short alias.

    Returns
    -------
    df_person : pd.DataFrame
        Wide format. Columns: new_id,
        {model_short}_alignment, n_q_{model_short}  (one pair per model).

    df_pq : pd.DataFrame
        Wide format. Columns: new_id,
        {model_short}_{question_var}  (NULL / NaN if not answered).
        Suitable for direct insertion into per_question_{q_set} DB table.
    """
    question_vars = questions_df["var_name"].tolist()
    # Keep only questions that actually exist as columns in the ESS data
    question_vars = [q for q in question_vars if q in df_ess.columns]

    models_in_votes = set(df_votes["model"].unique())
    active_models = [m for m in model_registry if m in models_in_votes]

    print(f"  Questions available in ESS: {len(question_vars)}")
    print(f"  Models with votes:          {len(active_models)}")
    print(f"  Respondents:                {len(df_ess):,}")

    # Pre-build lookup: (model, question_var) → (vote_answer, min_s, max_s)
    vote_lookup: dict[tuple, tuple] = {}
    relevant_votes = df_votes[df_votes["question_var"].isin(question_vars)]
    for row in relevant_votes.itertuples(index=False):
        vote_lookup[(row.model, row.question_var)] = (
            float(row.majority_vote_answer),
            float(row.min_scale),
            float(row.max_scale),
        )

    # --- Initialise output columns ---
    # Columns are accumulated in dicts and assembled once at the end; adding
    # them one at a time to a live DataFrame fragments it (and is slow, since
    # there is one per model × question).
    new_ids = df_ess["new_id"].values
    person_cols: dict[str, np.ndarray] = {}
    pq_cols: dict[str, np.ndarray] = {}

    # Pre-convert all question columns to float arrays once
    answer_arrays: dict[str, np.ndarray] = {
        q: pd.to_numeric(df_ess[q], errors="coerce").values
        for q in question_vars
    }

    for model_full in active_models:
        ms = model_registry[model_full]

        # group_key → list of per-question score arrays (N,)
        group_score_lists: dict[str, list] = {}

        for q in question_vars:
            key = (model_full, q)
            if key not in vote_lookup:
                continue

            vote_ans, min_s, max_s = vote_lookup[key]
            scale_range = max_s - min_s
            if scale_range == 0:
                continue

            answers = answer_arrays[q]
            valid = (answers >= min_s) & (answers <= max_s) & ~np.isnan(answers)
            pq_scores = np.where(
                valid,
                1.0 - np.abs(answers - vote_ans) / scale_range,
                np.nan,
            )

            # Per-question column in df_pq (NaN → stored as float NaN)
            pq_cols[f"{ms}_{q}"] = pq_scores

            group_key = question_group_map.get(q, q)
            group_score_lists.setdefault(group_key, []).append(pq_scores)

        if not group_score_lists:
            continue

        # --- Average within each question group ---
        # All-NaN rows are masked out by any_valid; silence nanmean's
        # "Mean of empty slice" warning on them.
        group_means: list[np.ndarray] = []
        for arrays in group_score_lists.values():
            stacked = np.column_stack(arrays)          # (N, n_sub)
            any_valid = ~np.all(np.isnan(stacked), axis=1)  # (N,)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                gm = np.nanmean(stacked, axis=1)       # (N,)
            group_means.append(np.where(any_valid, gm, np.nan))

        # --- Person-level: mean across question groups ---
        all_groups = np.column_stack(group_means)      # (N, n_groups)
        n_q = np.sum(~np.isnan(all_groups), axis=1).astype(float)
        any_group_valid = n_q > 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            person_align = np.nanmean(all_groups, axis=1)  # (N,)
        person_align = np.where(any_group_valid, person_align, np.nan)

        n_q = np.where(any_group_valid, n_q, np.nan)

        person_cols[f"{ms}_alignment"] = person_align
        person_cols[f"n_q_{ms}"] = n_q

    df_person = pd.DataFrame({"new_id": new_ids, **person_cols})
    df_pq = pd.DataFrame({"new_id": new_ids, **pq_cols})

    print(
        f"  Person-alignment table: {df_person.shape[0]:,} rows, "
        f"{df_person.shape[1]} cols"
    )
    print(f"  Per-question table:     {df_pq.shape[1] - 1} score columns")

    return df_person, df_pq
