# LLMs x ESS — Value Alignment Analysis

Analysis pipeline measuring how well LLM answers to value-laden
survey questions align with human responses from the European Social Survey
(ESS), broken down by country and socio-demographic group. The pipeline reproduces 
the results from the paper "People Are Not Just Their Countries. Disentangling Social Determinants of LLM Value Alignment Across Europe", published in the [Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES 2026)](https://doi.org/10.1609/aies.v9i4.43344). The repository also includes the Supplementary Material pdf for the paper.

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout

```
run_pipeline.py   CLI entry point — see "Running the pipeline" below
pipeline/         All analysis code (alignment, bootstrap, plots, IPW, ...)
data/             Question metadata and model answers; the raw ESS data and
                  the SQLite DBs also live here but are not included (see "Data")
prompts/          Prompts sent to the models (the model-calling code itself
                  is not included — see "Prompts" below)
results/          Pipeline output (tables + figures), one subfolder per
                  question set
```

## Data

### 1. ESS survey data

Download the ESS Round 11 zip file "ESS11e04_0.zip" from the
[European Social Survey data portal](https://ess.sikt.no/) and extract it into
`data/`, so that both of these exist:

```
data/ESS11e04_0/ESS11e04_0.csv
data/ESS11e04_0/ESS11e04_0 codebook.html
```

Build the SQLite database from it:

```bash
python run_pipeline.py --data-preparation
```

This creates `data/ESS11.db` with an `ess11` table (one row per respondent), and a `demographics` table.
Variable names throughout the pipeline match the official ESS variable names.

### 2. Value-laden questions

`data/value_laden_Qs.csv` lists every value-laden survey question used in the
analysis, together with its topic, response scale, and answer labels.

### 3. Model answers

The raw model answers are not included in this repository but can be requested
from the authors. They come as a SQLite database with an `interactions` table;
place it at `data/model_responses.db` (or point `--model-responses-db-path` at
it). It is used to generate
`data/model_majority_votes.csv`, which contains the majority-vote answer per
model/question as well as every individual answer (not just the majority vote),
plus `data/model_refusal_stats.csv` and `data/model_answer_variation_stats.csv`.
These three files are included in the `data/` folder.

Because they are included in the repository, the raw answers are only needed to
regenerate them. If `data/model_responses.db` is not present, the majority-vote
extraction step is skipped automatically and the pipeline runs from the
included CSVs (`--skip-majority-votes` forces this even when the DB is present).

## Running the pipeline

See `python run_pipeline.py --list-question-sets` for the full list of valid
`--question-set` values (`all`, `all_answered`, `just_values`, or any topic, e.g.
`trust`).

To obtain the core results from the paper, run the following in order
(`--data-plots` combines the outputs of the earlier runs, so run it last):

```bash
# Alignment scores (with bootstrap confidence intervals) + figures
python run_pipeline.py --question-set all_answered

# Predictive modeling: alignment vs. socio-demographics (LR / GBM)
python run_pipeline.py --question-set all_answered --run-method-comp

# Inverse propensity weighting (reweight countries on socio-demographics)
python run_pipeline.py --question-set all_answered --run-ipw

# Alignment scores (with bootstrap confidence intervals) + figures; all questions and PVQ subset
python run_pipeline.py --question-set all
python run_pipeline.py --question-set just_values --run-method-comp

# The comparison plot of the predictive modelling + some of the plots from the supplementary material
python run_pipeline.py --data-plots
```

Each stage can be skipped with `--skip-alignment`, `--skip-bootstrap`, or
`--skip-plots` if it has already been run. Run `python run_pipeline.py --help`
for the full set of options.

## Supplementary calculations

To reproduce the results given in the Supplementary Material run the following:

```bash
# Response extremeness vs. alignment
python run_pipeline.py --data-plots

# Joint (double) bootstrap over model answers + population resampling
python run_pipeline.py --question-set all_answered --run-double-bootstrap

# Midpoint-answer robustness check
python run_pipeline.py --question-set all_answered --run-midpoint-robustness
```

## Prompts

The exact prompts sent to each model are constructed in `prompts/`. The code that issues
the model calls is not included — answers were collected via the batch APIs where available, and the standard chat API otherwise.

## Citation


```bibtex
@inproceedings{wightman2026people,
  title     = {People Are Not Just Their Countries. Disentangling Social
               Determinants of {LLM} Value Alignment Across Europe},
  author    = {Wightman, Maria-Louisa and Bied, Guillaume and De Bie, Tijl},
  booktitle = {Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society},
  volume    = {9},
  number    = {4},
  pages     = {2725--2737},
  year      = {2026},
  doi       = {10.1609/aies.v9i4.43344},
  url       = {https://ojs.aaai.org/index.php/AIES/article/view/43344}
}
```


