"""
Pipeline configuration: model registry, variable lists, and runtime defaults.
Edit MODEL_REGISTRY here when new models are added.
"""
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # repository root
DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "ESS11.db"
DEFAULT_MODEL_RESPONSES_DB_PATH = DATA_DIR / "model_responses.db"
DEFAULT_QUESTIONS_CSV = DATA_DIR / "value_laden_Qs.csv"
DEFAULT_MAJORITY_VOTES_CSVS = [
    DATA_DIR / "model_majority_votes.csv",
]
DEFAULT_RAW_ESS11_CSV = DATA_DIR / "ESS11e04_0" / "ESS11e04_0.csv"
DEFAULT_RAW_CODEBOOK_HTML = DATA_DIR / "ESS11e04_0" / "ESS11e04_0 codebook.html"
RESULTS_DIR = BASE_DIR / "results"

# ── Bootstrap defaults ────────────────────────────────────────────────────────
N_BOOTSTRAP = 5000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42
N_PROCESSES = None    # None → 80 % of available cores
BATCH_SIZE = 25

# ── Model registry ─────────────────────────────────────────────────────────────
# Maps the exact model string in majority_votes CSV → short DB-safe alias.
# Short alias is used in DB column names and output file names.
# Add new models here; the pipeline warns and skips any model not listed.
MODEL_REGISTRY: dict[str, str] = {
    # OpenAI
    "gpt-5.5-2026-04-23":          "gpt5_5",
    "gpt-5.2-2025-12-11":          "gpt5_2",
    # Anthropic
    "claude-opus-4-6":             "claude_opus46",
    "claude-opus-4-7":             "claude_opus47",
    "claude-sonnet-4-5-20250929":  "claude_s45",
    # DeepSeek
    "deepseek_v4_pro":             "deepseek_v4",
    "deepseek_reasoner":           "deepseek_v3",
    # Mistral
    "mistral-large-latest":        "mistral_lg",
    "mistral-medium-3.5":          "mistral_m35_hg",
    "mistral-medium-3.5-medium":   "mistral_m35_md",
}

# ── Question group map ────────────────────────────────────────────────────────
# Sub-questions given to different random sub-samples of respondents.
# Within each group, scores are averaged before the person-level mean is taken,
# so they count as a single question in the final alignment score.
QUESTION_GROUP_MAP: dict[str, str] = {
    "testjc34": "testjc34_37_40",
    "testjc37": "testjc34_37_40",
    "testjc40": "testjc34_37_40",
    "testjc35": "testjc35_38_41",
    "testjc38": "testjc35_38_41",
    "testjc41": "testjc35_38_41",
    "testjc36": "testjc36_39_42",
    "testjc39": "testjc36_39_42",
    "testjc42": "testjc36_39_42",
}

# ── Socio-demographic variables ───────────────────────────────────────────────
# Variables with (*) are derived on-the-fly from ESS raw columns and may be
# absent if the source column is not in the DB; they are skipped gracefully.
SD_VARIABLES: list[str] = [
    "gender_clean",           # nobingnd
    "ethnicmajority_clean",   # feethngr
    "immigration_background", # brncntr + cntbrthd/fbrncntc/mbrncntc
    "education_bracket",      # edulvlb → ISCED brackets
    "income_clean",           # hinctnta (1-10)
    "generations",            # agea → 5 cohorts
    "religiosity_clean",      # rlgdgr → Low/Medium/High
    "domicil_clean",          # domicil (1-5, already in demographics)
    "political_interest_clean",  # polintr (1-4)
    "political_involvement",  # sum of 7 participation items
    "internet_time_bracket",  # netustm → time brackets
    "cntry",                  # direct from ess11
    "hincfel_clean",          # hincfel (1-4)
    "fnsdfml_clean",          # fnsdfml (1-5)
    "religion_clean",         # rlgdnm (*)
    "mnactic_clean",          # mnactic / mainact (*)
    "isco08_major_group",     # iscoco (*)
]

VARIABLE_LABELS: dict[str, str] = {
    "gender_clean":             "Gender",
    "ethnicmajority_clean":     "Ethnic Majority",
    "education_bracket":        "Education (ISCED)",
    "income_clean":             "Income Decile",
    "generations":              "Generations",
    "internet_time_bracket":    "Internet Time per Day",
    "religion_clean":           "Religious Denomination",
    "religiosity_clean":        "Religiosity Level",
    "domicil_clean":            "Domicile Type",
    "political_interest_clean": "Political Interest",
    "political_involvement":    "Political Involvement",
    "mnactic_clean":            "Main Activity",
    "isco08_major_group":       "Occupation (ISCO-08)",
    "cntry":                    "Country",
    "immigration_background":   "Immigration Background",
    "hincfel_clean":            "Household Income Feeling",
    "fnsdfml_clean":            "Childhood Financial Difficulties",
}

GROUP_LABELS: dict[str, dict] = {
    "education_bracket": {
        "0-1: ≤ Lower secondary":      "0-1: Lower secondary",
        "0-1: Less than lower secondary": "0-1: Lower secondary",
        "2: Lower secondary":           "2: Lower secondary",
        "3: Upper secondary":           "3: Upper secondary",
        "4: Post-secondary non-tertiary": "4: Post-second. non-tertiary",
        "5: Short-cycle tertiary":      "5: Short-cycle tertiary",
        "6: Bachelor":                  "6: Bachelor",
        "7: Master":                    "7: Master",
        "8: Doctoral":                  "8: Doctoral",
    },
    "gender_clean": {1.0: "Man", 2.0: "Woman", 3.0: "Other"},
    "ethnicmajority_clean": {1.0: "Ethnic majority", 2.0: "Ethnic minority"},
    "political_interest_clean": {
        1.0: "Very interested",
        2.0: "Quite interested",
        3.0: "Hardly interested",
        4.0: "Not at all interested",
    },
    "religiosity_clean": {
        "0-2: Low religiosity":    "Low (0-2)",
        "3-7: Medium religiosity": "Medium (3-7)",
        "8-10: High religiosity":  "High (8-10)",
    },
    "domicil_clean": {
        "A big city":                     "Big city",
        "Suburbs or outskirts of big city": "Suburbs",
        "Town or small city":              "Town or small city",
        "Country village":                 "Country village",
        "Farm or home in countryside":     "Farm or countryside",
    },
    "generations": {
        "Gen Z":             "Gen Z",
        "Millennials":       "Millennials",
        "Gen X":             "Gen X",
        "Baby Boomers":      "Baby Boomers",
        "Silent Generation": "Silent Generation",
    },
    "internet_time_bracket": {
        "0: No use":              "None",
        "1-30: Up to 1/2 hour":  "< 30 min",
        "31-60: 1/2-1 hour":     "30-60 min",
        "61-180: 1-3 hours":     "1-3 h",
        "181-300: 3-5 hours":    "3-5 h",
        "301-420: 5-7 hours":    "5-7 h",
        "421+: More than 7 hours": "> 7 h",
    },
    "mnactic_clean": {
        1.0: "Paid work",
        2.0: "Education",
        3.0: "Unemployed, seeking",
        4.0: "Unemployed, not seeking",
        5.0: "Permanently sick / disabled",
        6.0: "Retired",
        7.0: "Military / community service",
        8.0: "Housework / children",
        9.0: "Other",
    },
    "isco08_major_group": {
        0.0: "Armed forces",
        1.0: "Managers",
        2.0: "Professionals",
        3.0: "Technicians / Associates",
        4.0: "Clerical support workers",
        5.0: "Service and sales workers",
        6.0: "Agricultural workers",
        7.0: "Craft and trades workers",
        8.0: "Plant / machine operators",
        9.0: "Elementary occupations",
    },
    "religion_clean": {
        0.0: "No Religion",
        1.0: "Roman Catholic",
        2.0: "Protestant",
        3.0: "Eastern Orthodox",
        4.0: "Other Christian",
        5.0: "Jewish",
        6.0: "Islam",
        7.0: "Eastern religions",
        8.0: "Other non-Christian",
    },
    "immigration_background": {
        "No Migration Background":           "None",
        "Western Migration Background":      "Western",
        "Non-Western Migration Background":  "Non-Western",
        "Mixed Migration Background":        "Mixed",
    },
    "hincfel_clean": {
        1.0: "Living comfortably",
        2.0: "Coping",
        3.0: "Difficult",
        4.0: "Very difficult",
    },
    "fnsdfml_clean": {
        1.0: "Always",
        2.0: "Often",
        3.0: "Sometimes",
        4.0: "Hardly ever",
        5.0: "Never",
    },
}

# Explicit ordering for variables where natural sort is wrong
GROUP_ORDER: dict[str, list] = {
    "generations": [
        "Gen Z", "Millennials", "Gen X", "Baby Boomers", "Silent Generation"
    ],
    "domicil_clean": [
        "A big city",
        "Suburbs or outskirts of big city",
        "Town or small city",
        "Country village",
        "Farm or home in countryside",
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
    "political_interest_clean": [1.0, 2.0, 3.0, 4.0],
    "income_clean": list(range(1, 11)),
    "immigration_background": [
        "No Migration Background",
        "Western Migration Background",
        "Non-Western Migration Background",
        "Mixed Migration Background",
    ],
    "hincfel_clean": [1.0, 2.0, 3.0, 4.0],
    "fnsdfml_clean": [1.0, 2.0, 3.0, 4.0, 5.0],
}

# Colour palette — used across all plots 
COLOR_PALETTE: list[str] = [
    "#C8402A",  # 1  brick red
    "#8BBCD4",  # 2  steel blue
    "#C3D140",  # 3  yellow-green
    "#F29245",  # 4  orange
    "#9890C4",  # 5  lavender
    "#0A5B7C",  # 6  dark teal
    "#50ABA8",  # 7  medium teal
    "#276B50",  # 8  forest green
    "#E8B8CC",  # 9  light pink
    "#E9DA45",  # 10 yellow
]


# ISO-2 country code - human-readable English name.
# Covers the 30 countries present in ESS11 plus a few extras.
COUNTRY_NAME: dict[str, str] = {
    "AT": "Austria",     "BE": "Belgium",      "BG": "Bulgaria",
    "CH": "Switzerland", "CY": "Cyprus",       "CZ": "Czechia",
    "DE": "Germany",     "EE": "Estonia",      "ES": "Spain",
    "FI": "Finland",     "FR": "France",       "GB": "United Kingdom",
    "GR": "Greece",      "HR": "Croatia",      "HU": "Hungary",
    "IE": "Ireland",     "IL": "Israel",       "IS": "Iceland",
    "IT": "Italy",       "LT": "Lithuania",    "LV": "Latvia",
    "ME": "Montenegro",  "NL": "Netherlands",  "NO": "Norway",
    "PL": "Poland",      "PT": "Portugal",     "RS": "Serbia",
    "SE": "Sweden",      "SI": "Slovenia",     "SK": "Slovakia",
    "UA": "Ukraine",
}
