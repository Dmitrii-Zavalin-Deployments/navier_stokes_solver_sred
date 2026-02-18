# tests/schema/test_step1_schema_vs_final_schema.py

import json
from pathlib import Path
from tests.helpers.solver_step1_output_schema import EXPECTED_STEP1_SCHEMA

FINAL_SCHEMA_PATH = Path(__file__).parents[2] / "schema" / "solver_output_schema.json"

with FINAL_SCHEMA_PATH.open() as f:
    FINAL_SCHEMA = json.load(f)

FINAL_KEYS = set(FINAL_SCHEMA["properties"].keys())


def test_step1_schema_is_subset_of_final_schema():
    """
    Ensures Step 1 output is structurally compatible with the final solver output.
    """
    for key in EXPECTED_STEP1_SCHEMA:
        assert key in FINAL_KEYS, f"Missing in final schema: {key}"
