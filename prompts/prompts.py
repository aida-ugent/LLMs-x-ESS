from pathlib import Path

import pandas as pd
import random

SCALE_LEVELS_CSV = Path(__file__).resolve().parent / "scale_levels.csv"

class PromptBuilder:
    # def __init__(self, system_role="You are a helpful assistant."):
    def __init__(self, system_role=None):
        self.system_role = system_role
        self.scale_levels = pd.read_csv(SCALE_LEVELS_CSV, encoding="utf-8", na_filter=False)

    def make_likert_prompt(self, var_name, question, preamble, min_scale, min_label, max_scale, max_label, include_preamble = True):

        min_scale = int(float(min_scale))
        max_scale = int(float(max_scale))

        if not include_preamble:
            preamble = ''

        if not preamble:
            preamble = ''

        likert_scale_structure = f"""Answer with a whole number on the scale of {min_scale} to {max_scale}, where {min_scale} means '{min_label}' and {max_scale} means '{max_label}'."""

        if preamble:
            return f"""{preamble}
{question.strip()}

{likert_scale_structure}
""".strip()
        else:
            return f"""{question.strip()}

{likert_scale_structure}
""".strip()



