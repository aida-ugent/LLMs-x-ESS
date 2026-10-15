"""
run_pipeline.py – CLI entry point for the question-set analysis pipeline.

Usage
-----
# List all valid question sets
python run_pipeline.py --list-question-sets

# Full pipeline for all questions
python run_pipeline.py --question-set all

# Single topic (e.g. trust) with fast bootstrap for testing
python run_pipeline.py --question-set trust --n-bootstrap 10

# Skip alignment calculation (e.g. already done) and only re-run plots
python run_pipeline.py --question-set trust --skip-alignment --skip-bootstrap

# Full pipeline including method comparison
python run_pipeline.py --question-set just_values --run-method-comp

Paths default to the repository layout (data/, results/); pass --db-path,
--questions-csv or --votes-csv explicitly to read from elsewhere.
"""

import argparse
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent

# Allow running from anywhere by adding the script's directory to sys.path
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline.config import (
    DEFAULT_DB_PATH, DEFAULT_MODEL_RESPONSES_DB_PATH,
    DEFAULT_MAJORITY_VOTES_CSVS, DEFAULT_QUESTIONS_CSV,
    DEFAULT_RAW_ESS11_CSV, DEFAULT_RAW_CODEBOOK_HTML,
    MODEL_REGISTRY, N_BOOTSTRAP, CONFIDENCE_LEVEL, RANDOM_SEED,
    N_PROCESSES, BATCH_SIZE, RESULTS_DIR, SD_VARIABLES,
)
from pipeline.data import (
    list_question_sets, load_ess_data, load_majority_votes,
    load_questions, q_set_key,
    write_alignments_to_db, write_per_question_to_db,
)
from pipeline.alignment import calculate_alignments
from pipeline.bootstrap import run_bootstrap
from pipeline.plots import (
    plot_countries, plot_heatmap_all_models, plot_heatmap_cross_model_mean,
    plot_heatmap_cross_model_overall, plot_heatmap_cross_model_combined,
    plot_heatmap_by_topic, plot_master_overview,
    plot_all_models_comparison,
    plot_model_correlation_heatmap,
    plot_model_correlation_heatmap_person,
    plot_model_correlation_heatmap_combined,
    plot_cross_model_mean_deviation,
    plot_overall_alignment_comparison,
)
from pipeline.data_plots import run_data_plots
from pipeline.model_summary_table import run_model_summary_table
from pipeline.extremeness_plots import run_extremeness_plots


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Question-set alignment analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--question-set", "-q",
        default=None,
        help=(
            "Question set to analyse. One of: 'all', 'all_answered', "
            "'just_values', 'non_value', or any topic string. "
            "Use --list-question-sets to see all options."
        ),
    )
    p.add_argument(
        "--list-question-sets", "-l",
        action="store_true",
        help="Print all available question sets and exit.",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to ESS11.db (default: {DEFAULT_DB_PATH})",
    )
    p.add_argument(
        "--data-preparation",
        action="store_true",
        help=(
            "Build the `ess11` and `demographics` tables in ESS11.db from the "
            "official ESS11e04_0 export (CSV + codebook HTML). Does not require "
            "--question-set. Rebuilding `demographics` discards any alignment "
            "columns it held; Stage 1 recomputes them. Other tables "
            "(per_question_*, …) are left untouched."
        ),
    )
    p.add_argument(
        "--raw-ess11-csv",
        type=Path,
        default=DEFAULT_RAW_ESS11_CSV,
        help=f"Path to ESS11e04_0.csv (default: {DEFAULT_RAW_ESS11_CSV})",
    )
    p.add_argument(
        "--raw-codebook-html",
        type=Path,
        default=DEFAULT_RAW_CODEBOOK_HTML,
        help=(
            "Path to the ESS11 codebook HTML, used for the demographics value "
            f"labels (default: {DEFAULT_RAW_CODEBOOK_HTML})"
        ),
    )
    p.add_argument(
        "--model-responses-db-path",
        type=Path,
        default=DEFAULT_MODEL_RESPONSES_DB_PATH,
        help=(
            "Path to the raw model-responses SQLite DB "
            f"(default: {DEFAULT_MODEL_RESPONSES_DB_PATH})"
        ),
    )
    p.add_argument(
        "--votes-csv",
        nargs="+",
        type=Path,
        default=DEFAULT_MAJORITY_VOTES_CSVS,
        help="Path(s) to model_majority_votes.csv file(s).",
    )
    p.add_argument(
        "--questions-csv",
        type=Path,
        default=DEFAULT_QUESTIONS_CSV,
        help=f"Path to value_laden_Qs.csv (default: {DEFAULT_QUESTIONS_CSV})",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help=f"Number of bootstrap samples (default: {N_BOOTSTRAP})",
    )
    p.add_argument(
        "--n-processes",
        type=int,
        default=None,
        help="Parallel processes for bootstrap (default: 80%% of cores)",
    )
    p.add_argument(
        "--run-method-comp",
        action="store_true",
        help="Also run the method comparison analysis.",
    )
    p.add_argument(
        "--run-ipw",
        action="store_true",
        help=(
            "Also run the IPW (inverse propensity weighting) analysis: "
            "reweight country samples on sociodemographics, compute "
            "between-country variance pre/post with bootstrap CIs, "
            "and produce country-mean + before-after plots."
        ),
    )
    p.add_argument(
        "--ipw-n-bootstrap",
        type=int,
        default=100,
        help=(
            "Bootstrap replicates for the IPW stage (default: 100). "
            "Each replicate refits cross-fitted propensity, so this is slow."
        ),
    )
    p.add_argument(
        "--ipw-crossfit-folds",
        type=int,
        default=5,
        help="Cross-fitting folds for IPW propensity (default: 5).",
    )
    p.add_argument(
        "--ipw-skip-bootstrap",
        action="store_true",
        help=(
            "Skip the bootstrap stage of the IPW analysis (point estimates only)."
        ),
    )
    p.add_argument(
        "--weight-col",
        default="pspwght",
        choices=["pspwght", "anweight"],
        help="Survey weight column for method comparison (default: pspwght)",
    )
    p.add_argument(
        "--skip-majority-votes",
        action="store_true",
        help=(
            "Skip majority-vote extraction (use existing model_majority_votes.csv). "
            "Skipped automatically if --model-responses-db-path does not exist."
        ),
    )
    p.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Skip alignment calculation (use existing DB columns).",
    )
    p.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip bootstrap analysis (use existing CSV files).",
    )
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    p.add_argument(
        "--run-double-bootstrap",
        action="store_true",
        help=(
            "Run the double bootstrap: joint model-answer + population "
            "resampling.  Produces the same CSVs and plots as the standard "
            "bootstrap but captures both sources of uncertainty."
        ),
    )
    p.add_argument(
        "--double-bootstrap-plots-only",
        action="store_true",
        help=(
            "Regenerate double-bootstrap plots from existing CSV results "
            "without re-running the bootstrap computation.  "
            "Requires --question-set."
        ),
    )
    p.add_argument(
        "--double-bootstrap-n",
        type=int,
        default=None,
        help=(
            "Bootstrap iterations for the double bootstrap "
            "(default: same as --n-bootstrap)."
        ),
    )
    p.add_argument(
        "--regen-comparison-plots",
        action="store_true",
        help=(
            "Regenerate the all-models comparison plot for every question set "
            "that already has bootstrap results in the results directory. "
            "Does not require --question-set."
        ),
    )
    p.add_argument(
        "--data-plots",
        action="store_true",
        help=(
            "Generate descriptive data plots (people non-answer rates, model "
            "refusal heatmap, questions-answered histogram), the per-model "
            "summary table, and the extremeness plots. "
            "Does not require --question-set; uses the full 'all' question set. "
            "Output goes to results/data_plots/."
        ),
    )
    p.add_argument(
        "--extremeness-qset",
        default="all_answered",
        metavar="QSET",
        help=(
            "Question set used to compute response extremeness when "
            "--data-plots is active (default: all_answered)."
        ),
    )
    p.add_argument(
        "--regen-ipw-plots",
        action="store_true",
        help=(
            "Regenerate IPW figures from previously saved CSVs without "
            "re-fitting any models. Requires --question-set (or scans all "
            "question sets with existing IPW results if omitted)."
        ),
    )
    p.add_argument(
        "--run-midpoint-robustness",
        action="store_true",
        help=(
            "Run the midpoint robustness check: compute alignment for a "
            "synthetic model that always answers the centre of each scale, "
            "run bootstrap, and produce the master overview plot. "
            "Results are written under {results_dir}/{qset}/midpoint_robustness/."
        ),
    )
    p.add_argument(
        "--midpoint-skip-alignment",
        action="store_true",
        help="Skip midpoint alignment calculation (column already in DB).",
    )
    p.add_argument(
        "--midpoint-skip-bootstrap",
        action="store_true",
        help="Skip midpoint bootstrap (use existing CSV files).",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def _stage_data_preparation(args) -> None:
    from pipeline.data_preparation import (
        build_demographics_table, build_ess11_db,
    )

    build_ess11_db(raw_csv_path=args.raw_ess11_csv, db_path=args.db_path)
    build_demographics_table(
        db_path            = args.db_path,
        codebook_html_path = args.raw_codebook_html,
    )


def _stage_majority_votes(args) -> None:
    from pipeline.majority_votes import extract_majority_votes

    # Output goes to DATA_DIR so Stage 1 picks it up via --votes-csv default
    from pipeline.config import DATA_DIR
    extract_majority_votes(
        model_responses_db_path = args.model_responses_db_path,
        questions_csv = args.questions_csv,
        output_dir    = DATA_DIR,
    )


def _stage_alignment(args, qset_key: str, questions_df, output_root: Path) -> None:
    print("\n" + "=" * 60)
    print("STAGE 1 – Alignment calculation")
    print("=" * 60)

    df_ess   = load_ess_data(args.db_path)
    df_votes = load_majority_votes(args.votes_csv, MODEL_REGISTRY)

    df_person, df_pq = calculate_alignments(
        df_ess, df_votes, questions_df,
    )

    print("\nWriting to database …")
    write_alignments_to_db(args.db_path, df_person, qset_key)
    write_per_question_to_db(args.db_path, df_pq, qset_key)

    # CSV backup only for the full 'all' question set
    if qset_key == "all":
        pq_backup = output_root / f"per_question_{qset_key}.csv"
        df_pq.to_csv(pq_backup, index=False)
        print(f"  Per-question CSV backup: {pq_backup}")


def _stage_bootstrap(args, qset_key: str, output_root: Path) -> Path:
    print("\n" + "=" * 60)
    print("STAGE 2 – Bootstrap analysis")
    print("=" * 60)

    boot_dir = output_root / "bootstrap_results"
    run_bootstrap(
        db_path          = args.db_path,
        question_set_key = qset_key,
        model_registry   = MODEL_REGISTRY,
        sd_variables     = SD_VARIABLES,
        output_dir       = boot_dir,
        n_bootstrap      = args.n_bootstrap,
        conf_level       = CONFIDENCE_LEVEL,
        seed             = RANDOM_SEED,
        n_processes      = args.n_processes or N_PROCESSES,
        batch_size       = BATCH_SIZE,
    )
    return boot_dir


def _stage_plots(
    args, qset_key: str, questions_df, output_root: Path, boot_dir: Path
) -> None:
    print("\n" + "=" * 60)
    print("STAGE 3 – Plots")
    print("=" * 60)

    figs_boot    = output_root / "figures" / "bootstrap"
    figs_heatmap = output_root / "figures" / "heatmaps"
    figs_boot.mkdir(parents=True, exist_ok=True)
    figs_heatmap.mkdir(parents=True, exist_ok=True)

    # Determine which model_shorts are present in the bootstrap results
    import pandas as _pd
    _overall_csv = boot_dir / "alignment_overall_population.csv"
    if _overall_csv.exists():
        _active_models = sorted(_pd.read_csv(_overall_csv)["model"].unique().tolist())
    else:
        _active_models = list(MODEL_REGISTRY.values())

    # # 3a. Master overview – one figure per model
    # print("\n  [3a] Master overview (per model) …")
    # for ms in _active_models:
    #     plot_master_overview(boot_dir, figs_boot, ms, question_set_key=qset_key)

    # # 3b. Countries – one figure per model
    # print("  [3b] Countries CI plot (per model) …")
    # for ms in _active_models:
    #     plot_countries(boot_dir, figs_boot, ms, question_set_key=qset_key)

    # # 3c. Heatmaps by topic (SD groups) – ONE figure per model
    # print("  [3c] Heatmaps by topic (SD groups) …")
    # for model_full, model_short in MODEL_REGISTRY.items():
    #     plot_heatmap_by_topic(
    #         db_path          = args.db_path,
    #         question_set_key = qset_key,
    #         model_short      = model_short,
    #         questions_df     = questions_df,
    #         output_dir       = figs_heatmap,
    #         mode             = "sd_groups",
    #     )

    # # 3d. Heatmaps by topic (countries) – ONE figure per model
    # print("  [3d] Heatmaps by topic (countries) …")
    # for model_full, model_short in MODEL_REGISTRY.items():
    #     plot_heatmap_by_topic(
    #         db_path          = args.db_path,
    #         question_set_key = qset_key,
    #         model_short      = model_short,
    #         questions_df     = questions_df,
    #         output_dir       = figs_heatmap,
    #         mode             = "countries",
    #     )

    # # 3e. All-models alignment-difference heatmap (one per model)
    # print("  [3e] All-models difference heatmap …")
    # plot_heatmap_all_models(
    #     db_path          = args.db_path,
    #     question_set_key = qset_key,
    #     model_registry   = MODEL_REGISTRY,
    #     output_dir       = figs_heatmap,
    # )

    # 3e2. Cross-model mean deviation heatmap (single plot, mean across models)
    print("  [3e2] Cross-model mean deviation heatmap …")
    plot_heatmap_cross_model_mean(
        db_path          = args.db_path,
        question_set_key = qset_key,
        model_registry   = MODEL_REGISTRY,
        output_dir       = figs_heatmap,
    )

    # 3e3. Cross-model overall-mean deviation heatmap
    print("  [3e3] Cross-model overall-mean deviation heatmap …")
    plot_heatmap_cross_model_overall(
        db_path          = args.db_path,
        question_set_key = qset_key,
        model_registry   = MODEL_REGISTRY,
        output_dir       = figs_heatmap,
    )

    # 3e4. Combined side-by-side: country-mean | overall-mean deviation
    print("  [3e4] Cross-model combined deviation heatmap …")
    plot_heatmap_cross_model_combined(
        db_path          = args.db_path,
        question_set_key = qset_key,
        model_registry   = MODEL_REGISTRY,
        output_dir       = figs_heatmap,
    )

    # 3f. All-models comparison (SD groups + countries, all models overlaid)
    print("  [3f] All-models comparison plot …")
    plot_all_models_comparison(boot_dir, figs_boot, question_set_key=qset_key)

    # # 3g. All-models by group (models as rows, groups as scattered dots) 
    # print("  [3g] All-models by-group plot …")
    # plot_all_models_by_group(boot_dir, figs_boot, question_set_key=qset_key)

    # 3h. Pairwise model correlation heatmap (question-level mean alignment)
    print("  [3h] Pairwise model correlation heatmap (question-level mean alignment) …")
    plot_model_correlation_heatmap(
        db_path          = args.db_path,
        question_set_key = qset_key,
        model_registry   = MODEL_REGISTRY,
        output_dir       = figs_heatmap,
    )

    # 3i. Pairwise model correlation heatmap (person-level alignment scores)
    print("  [3i] Pairwise model correlation heatmap (person-level alignment scores) …")
    plot_model_correlation_heatmap_person(
        db_path          = args.db_path,
        question_set_key = qset_key,
        model_registry   = MODEL_REGISTRY,
        output_dir       = figs_heatmap,
    )

    # 3i2. Combined correlation heatmap (upper=majority-vote answers, lower=person-level alignment)
    print("  [3i2] Combined correlation heatmap (upper=majority-vote answers, lower=person-level alignment) …")
    plot_model_correlation_heatmap_combined(
        db_path              = args.db_path,
        question_set_key     = qset_key,
        model_registry       = MODEL_REGISTRY,
        output_dir           = figs_heatmap,
        majority_votes_csvs  = args.votes_csv,
    )

    # 3j. Cross-model mean deviation plots (3 variants)
    print("  [3j] Cross-model mean deviation plots …")
    plot_cross_model_mean_deviation(
        bootstrap_results_dir = boot_dir,
        output_dir            = figs_boot,
        question_set_key      = qset_key,
    )


def _stage_data_plots(args) -> None:
    print("\n" + "=" * 60)
    print("STAGE – Data description plots")
    print("=" * 60)

    from pipeline.config import DATA_DIR
    refusal_csv = DATA_DIR / "model_refusal_stats.csv"
    if not refusal_csv.exists():
        print(
            f"  WARNING: {refusal_csv} not found. "
            "Run Stage 0 (majority-vote extraction) first to generate it."
        )
        return

    output_dir = RESULTS_DIR / "data_plots"
    run_data_plots(
        db_path              = args.db_path,
        refusal_stats_csv    = refusal_csv,
        questions_csv        = args.questions_csv,
        output_dir           = output_dir,
        majority_votes_csvs  = args.votes_csv,
    )

    print("\n  [Summary table] Per-model statistics …")
    run_model_summary_table(
        votes_csvs        = args.votes_csv,
        refusal_stats_csv = refusal_csv,
        questions_csv     = args.questions_csv,
        results_dir       = RESULTS_DIR,
        output_dir        = output_dir,
    )

    print("\n  [Extremeness plots] Alignment vs response extremeness …")
    run_extremeness_plots(
        db_path       = args.db_path,
        questions_csv = args.questions_csv,
        output_dir    = output_dir,
        question_set  = args.extremeness_qset,
    )


def _stage_method_comp(args, qset_key: str, output_root: Path) -> None:
    print("\n" + "=" * 60)
    print("STAGE 4 – Method comparison")
    print("=" * 60)

    from pipeline.method_comp import run_method_comp

    mc_dir = output_root / "method_comp"
    run_method_comp(
        db_path          = args.db_path,
        question_set_key = qset_key,
        model_registry   = MODEL_REGISTRY,
        output_dir       = mc_dir,
        weight_col       = args.weight_col,
    )


def _next_ipw_dir(output_root: Path) -> Path:
    """Return output_root/ipw if it doesn't exist, else output_root/ipw_1, ipw_2, …"""
    base = output_root / "ipw"
    if not base.exists():
        return base
    n = 1
    while (output_root / f"ipw_{n}").exists():
        n += 1
    return output_root / f"ipw_{n}"


def _stage_ipw(args, qset_key: str, output_root: Path) -> None:
    """
    Run the IPW (inverse propensity weighting) analysis for one qset.

    `just_values` is excluded from IPW (other stages still run on it).
    If the ipw/ output directory already exists a versioned sibling is used
    (ipw_1/, ipw_2/, …) so previous results are never overwritten.
    """
    if qset_key == "just_values":
        print("\n[skip] IPW: just_values is excluded from IPW by design.")
        return

    print("\n" + "=" * 60)
    print("STAGE 5 – Inverse Propensity Weighting (IPW)")
    print("=" * 60)

    from pipeline.ipw import run_ipw

    ipw_dir = _next_ipw_dir(output_root)
    if ipw_dir.name != "ipw":
        print(f"  [note] ipw/ already exists — writing to {ipw_dir.name}/")

    run_ipw(
        db_path          = args.db_path,
        question_set_key = qset_key,
        output_dir       = ipw_dir,
        model_registry   = MODEL_REGISTRY,
        weight_col       = args.weight_col,
        n_bootstrap      = args.ipw_n_bootstrap,
        crossfit_folds   = args.ipw_crossfit_folds,
        skip_bootstrap   = args.ipw_skip_bootstrap,
        n_jobs           = -1,
    )


def _stage_regen_ipw_plots(args) -> None:
    from pipeline.ipw import regen_ipw_plots

    # Single question set
    if args.question_set is not None:
        from pipeline.data import q_set_key
        qset_key   = q_set_key(args.question_set)
        qset_dir   = RESULTS_DIR / qset_key
        # Find all ipw* subdirs for this qset
        ipw_dirs = sorted(d for d in qset_dir.glob("ipw*") if d.is_dir())
        if not ipw_dirs:
            print(f"  No IPW results found under {qset_dir}")
            return
        for ipw_dir in ipw_dirs:
            print(f"\n  Regenerating IPW plots: {ipw_dir}")
            regen_ipw_plots(ipw_dir)
        return

    # No question set → scan everything
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return
    found = 0
    for qset_dir in sorted(RESULTS_DIR.iterdir()):
        for ipw_dir in sorted(d for d in qset_dir.glob("ipw*") if d.is_dir()):
            meta = ipw_dir / "run_metadata.json"
            if not meta.exists():
                continue
            print(f"\n  Regenerating IPW plots: {ipw_dir}")
            regen_ipw_plots(ipw_dir)
            found += 1
    if found == 0:
        print("  No IPW results found.")
    else:
        print(f"\nDone — regenerated IPW plots for {found} directory/ies.")


def _stage_double_bootstrap_plots_only(
    args, qset_key: str, output_root: Path
) -> None:
    """Regenerate double-bootstrap plots from existing CSV results (no recompute)."""
    print("\n" + "=" * 60)
    print("STAGE – Double bootstrap plots (from existing results)")
    print("=" * 60)

    dboot_dir      = output_root / "double_bootstrap_results"
    scores_dir     = dboot_dir / "scores"
    deviations_dir = dboot_dir / "deviations"

    if not scores_dir.exists():
        print(f"  ERROR: scores directory not found: {scores_dir}")
        print("  Run --run-double-bootstrap first to generate the CSVs.")
        return

    import pandas as _pd
    _overall_csv = scores_dir / "alignment_overall_population.csv"
    if _overall_csv.exists():
        _active_models = sorted(
            _pd.read_csv(_overall_csv)["model"].unique().tolist()
        )
    else:
        _active_models = list(MODEL_REGISTRY.values())

    figs_scores = output_root / "figures" / "double_bootstrap" / "scores"
    figs_devs   = output_root / "figures" / "double_bootstrap" / "deviations"
    figs_scores.mkdir(parents=True, exist_ok=True)
    figs_devs.mkdir(parents=True, exist_ok=True)

    qset_label = f"{qset_key}_dboot"

    # print("\n  [DB-a] Master overview (per model) …")
    # for ms in _active_models:
    #     plot_master_overview(scores_dir, figs_scores, ms, question_set_key=qset_label)

    # print("  [DB-b] Countries CI plot (per model) …")
    # for ms in _active_models:
    #     plot_countries(scores_dir, figs_scores, ms, question_set_key=qset_label)

    # print("  [DB-c] All-models comparison plot …")
    # plot_all_models_comparison(scores_dir, figs_scores, question_set_key=qset_label)

    # print("  [DB-d] All-models by-group plot …")
    # plot_all_models_by_group(scores_dir, figs_scores, question_set_key=qset_label)

    print("  [DB-e] Cross-model mean deviation plots …")
    plot_cross_model_mean_deviation(
        bootstrap_results_dir=scores_dir,
        output_dir=figs_scores,
        question_set_key=qset_label,
        variants=("models",),
    )

    # print("  [DB-f] Per-model deviation overview (per model) …")
    # for ms in _active_models:
    #     plot_per_model_deviation_overview(
    #         deviations_dir, figs_devs, ms, question_set_key=qset_label
    #     )

    # print("  [DB-g] Per-model deviation countries (per model) …")
    # for ms in _active_models:
    #     plot_per_model_deviation_countries(
    #         deviations_dir, figs_devs, ms, question_set_key=qset_label
    #     )

    print("  [DB-h] Overall alignment comparison plot …")
    _overall_sources: list[tuple[str, Path]] = []
    _reg_all_answered = output_root / "bootstrap_results" / "alignment_overall_population.csv"
    if _reg_all_answered.exists():
        _overall_sources.append((f"Bootstrap ({qset_key})", _reg_all_answered))
    _reg_all = RESULTS_DIR / "all" / "bootstrap_results" / "alignment_overall_population.csv"
    if _reg_all.exists():
        _overall_sources.append(("Bootstrap (all)", _reg_all))
    _dboot_overall = scores_dir / "alignment_overall_population.csv"
    if _dboot_overall.exists():
        _overall_sources.append((f"Double bootstrap ({qset_key})", _dboot_overall))
    if _overall_sources:
        plot_overall_alignment_comparison(
            _overall_sources, figs_scores, question_set_key=qset_key
        )

    print(f"\n  Score plots   → {figs_scores}")
    print(f"  Deviation plots → {figs_devs}")


def _stage_double_bootstrap(
    args, qset_key: str, questions_df, output_root: Path
) -> None:
    print("\n" + "=" * 60)
    print("STAGE – Double bootstrap (model-answer + population)")
    print("=" * 60)

    from pipeline.double_bootstrap import run_double_bootstrap

    dboot_dir = output_root / "double_bootstrap_results"
    n_iter    = args.double_bootstrap_n or args.n_bootstrap

    run_double_bootstrap(
        votes_csvs         = args.votes_csv,
        db_path            = args.db_path,
        questions_df       = questions_df,
        model_registry     = MODEL_REGISTRY,
        sd_variables       = SD_VARIABLES,
        output_dir         = dboot_dir,
        n_bootstrap        = n_iter,
        conf_level         = CONFIDENCE_LEVEL,
        seed               = RANDOM_SEED,
        n_processes        = args.n_processes or N_PROCESSES,
        batch_size         = BATCH_SIZE,
    )

    if args.skip_plots:
        print("\n[skip] Double bootstrap plots")
        return

    scores_dir     = dboot_dir / "scores"
    deviations_dir = dboot_dir / "deviations"

    # Determine which model_shorts are present in the double bootstrap results
    import pandas as _pd
    _overall_csv = scores_dir / "alignment_overall_population.csv"
    if _overall_csv.exists():
        _active_models = sorted(
            _pd.read_csv(_overall_csv)["model"].unique().tolist()
        )
    else:
        _active_models = list(MODEL_REGISTRY.values())

    figs_scores = output_root / "figures" / "double_bootstrap" / "scores"
    figs_devs   = output_root / "figures" / "double_bootstrap" / "deviations"
    figs_scores.mkdir(parents=True, exist_ok=True)
    figs_devs.mkdir(parents=True, exist_ok=True)

    qset_label = f"{qset_key}_dboot"

    # print("\n  [DB-a] Master overview (per model) …")
    # for ms in _active_models:
    #     plot_master_overview(
    #         scores_dir, figs_scores, ms, question_set_key=qset_label
    #     )

    # print("  [DB-b] Countries CI plot (per model) …")
    # for ms in _active_models:
    #     plot_countries(
    #         scores_dir, figs_scores, ms, question_set_key=qset_label
    #     )

    # print("  [DB-c] All-models comparison plot …")
    # plot_all_models_comparison(
    #     scores_dir, figs_scores, question_set_key=qset_label
    # )

    # print("  [DB-d] All-models by-group plot …")
    # plot_all_models_by_group(
    #     scores_dir, figs_scores, question_set_key=qset_label
    # )

    print("  [DB-e] Cross-model mean deviation plots …")
    plot_cross_model_mean_deviation(
        bootstrap_results_dir = scores_dir,
        output_dir            = figs_scores,
        question_set_key      = qset_label,
        variants              = ("models",),
    )

    # print("  [DB-f] Per-model deviation overview (per model) …")
    # for ms in _active_models:
    #     plot_per_model_deviation_overview(
    #         deviations_dir, figs_devs, ms, question_set_key=qset_label
    #     )

    # print("  [DB-g] Per-model deviation countries (per model) …")
    # for ms in _active_models:
    #     plot_per_model_deviation_countries(
    #         deviations_dir, figs_devs, ms, question_set_key=qset_label
    #     )

    print("  [DB-h] Overall alignment comparison plot …")
    _overall_sources: list[tuple[str, Path]] = []
    _reg_all_answered = output_root / "bootstrap_results" / "alignment_overall_population.csv"
    if _reg_all_answered.exists():
        _overall_sources.append((f"Bootstrap ({qset_key})", _reg_all_answered))
    _reg_all = RESULTS_DIR / "all" / "bootstrap_results" / "alignment_overall_population.csv"
    if _reg_all.exists():
        _overall_sources.append(("Bootstrap (all)", _reg_all))
    _dboot_overall = scores_dir / "alignment_overall_population.csv"
    if _dboot_overall.exists():
        _overall_sources.append((f"Double bootstrap ({qset_key})", _dboot_overall))
    if _overall_sources:
        plot_overall_alignment_comparison(
            _overall_sources, figs_scores, question_set_key=qset_key
        )

    print(f"\n  Score plots   → {figs_scores}")
    print(f"  Deviation plots → {figs_devs}")


def _stage_midpoint_robustness(
    args, qset_key: str, questions_df, output_root: Path
) -> None:
    from pipeline.midpoint_robustness import run_midpoint_robustness

    midpoint_dir = output_root / "midpoint_robustness"
    figs_dir     = output_root / "figures" / "midpoint_robustness"
    run_midpoint_robustness(
        db_path          = args.db_path,
        question_set_key = qset_key,
        questions_df     = questions_df,
        output_dir       = midpoint_dir,
        figures_dir      = figs_dir,
        n_bootstrap      = args.n_bootstrap,
        n_processes      = args.n_processes or N_PROCESSES,
        skip_alignment   = args.midpoint_skip_alignment,
        skip_bootstrap   = args.midpoint_skip_bootstrap,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --data-preparation: build ess11 table from the raw CSV, standalone
    if args.data_preparation:
        _stage_data_preparation(args)
        return

    # --regen-ipw-plots: regenerate IPW figures from saved CSVs
    if args.regen_ipw_plots:
        _stage_regen_ipw_plots(args)
        return

    # --data-plots: standalone, no --question-set needed
    if args.data_plots:
        _stage_data_plots(args)
        return

    # --regen-comparison-plots: scan results dir, no --question-set needed
    if args.regen_comparison_plots:
        if not RESULTS_DIR.exists():
            print(f"Results directory not found: {RESULTS_DIR}")
            return
        found = 0
        for qset_dir in sorted(RESULTS_DIR.iterdir()):
            boot_dir = qset_dir / "bootstrap_results"
            if not (boot_dir / "alignment_overall_population.csv").exists():
                continue
            qset_key = qset_dir.name
            out_dir  = qset_dir / "figures" / "bootstrap"
            print(f"\n  Regenerating comparison plots: {qset_key!r}")
            plot_all_models_comparison(boot_dir, out_dir, question_set_key=qset_key)
            # plot_all_models_by_group(boot_dir, out_dir, question_set_key=qset_key)
            plot_model_correlation_heatmap(
                db_path          = args.db_path,
                question_set_key = qset_key,
                model_registry   = MODEL_REGISTRY,
                output_dir       = out_dir,
            )
            plot_model_correlation_heatmap_person(
                db_path          = args.db_path,
                question_set_key = qset_key,
                model_registry   = MODEL_REGISTRY,
                output_dir       = out_dir,
            )
            plot_model_correlation_heatmap_combined(
                db_path              = args.db_path,
                question_set_key     = qset_key,
                model_registry       = MODEL_REGISTRY,
                output_dir           = out_dir,
                majority_votes_csvs  = args.votes_csv,
            )
            figs_heatmap_regen = qset_dir / "figures" / "heatmaps"
            figs_heatmap_regen.mkdir(parents=True, exist_ok=True)
            plot_heatmap_cross_model_combined(
                db_path          = args.db_path,
                question_set_key = qset_key,
                model_registry   = MODEL_REGISTRY,
                output_dir       = figs_heatmap_regen,
            )
            plot_cross_model_mean_deviation(
                bootstrap_results_dir = boot_dir,
                output_dir            = out_dir,
                question_set_key      = qset_key,
            )
            # Regenerate method-comp combined plots if CSVs exist
            mc_dir = qset_dir / "method_comp"
            if mc_dir.exists() and any(mc_dir.glob("method_comp_*.csv")):
                from pipeline.method_comp import regen_method_comp_plots
                regen_method_comp_plots(mc_dir)
            found += 1
        if found == 0:
            print("  No question sets with bootstrap results found.")
        else:
            print(f"\nDone — regenerated comparison plots for {found} question set(s).")
        return

    # --list-question-sets
    if args.list_question_sets:
        sets = list_question_sets(args.questions_csv)
        print("Available question sets:")
        for s in sets:
            print(f"  {s}")
        return

    if args.question_set is None:
        parser.error("--question-set is required (or use --list-question-sets).")

    # Resolve key + output directory
    qset_key = q_set_key(args.question_set)
    output_root = RESULTS_DIR / qset_key
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\nPipeline started")
    print(f"  Question set : {args.question_set!r}  (key: {qset_key!r})")
    print(f"  DB           : {args.db_path}")
    print(f"  Results dir  : {output_root}")
    print(f"  N bootstrap  : {args.n_bootstrap}")

    # Load question metadata (needed by multiple stages)
    questions_df = load_questions(
        args.question_set, args.questions_csv,
        majority_votes_csvs=args.votes_csv,
        model_registry=MODEL_REGISTRY,
    )

    # Stage 0 – Majority votes
    # The raw model-responses DB is not distributed with the repository, so
    # fall back to the included majority-vote CSV when it is absent.
    if args.skip_majority_votes:
        print("\n[skip] Majority vote extraction")
    elif not args.model_responses_db_path.exists():
        print(
            f"\n[skip] Majority vote extraction – {args.model_responses_db_path} "
            "not found; using the existing majority-vote CSV(s)."
        )
    else:
        _stage_majority_votes(args)

    # Stage 1 – Alignment
    if not args.skip_alignment:
        _stage_alignment(args, qset_key, questions_df, output_root)
    else:
        print("\n[skip] Alignment calculation")

    # Stage 2 – Bootstrap
    boot_dir = output_root / "bootstrap_results"
    if not args.skip_bootstrap:
        boot_dir = _stage_bootstrap(args, qset_key, output_root)
    else:
        print("\n[skip] Bootstrap analysis")

    # Stage 3 – Plots
    if not args.skip_plots:
        _stage_plots(args, qset_key, questions_df, output_root, boot_dir)
    else:
        print("\n[skip] Plots")

    # Stage 4 – Method comparison (optional)
    if args.run_method_comp:
        _stage_method_comp(args, qset_key, output_root)

    # Stage 5 – IPW (optional)
    if args.run_ipw:
        _stage_ipw(args, qset_key, output_root)

    # Stage 6 – Double bootstrap (optional)
    if args.run_double_bootstrap:
        _stage_double_bootstrap(args, qset_key, questions_df, output_root)
    elif args.double_bootstrap_plots_only:
        _stage_double_bootstrap_plots_only(args, qset_key, output_root)

    # Stage 7 – Midpoint robustness check (optional)
    if args.run_midpoint_robustness:
        _stage_midpoint_robustness(args, qset_key, questions_df, output_root)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results in: {output_root}")


if __name__ == "__main__":
    main()
