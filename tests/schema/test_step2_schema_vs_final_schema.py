# tests/schema/test_step2_schema_vs_final_schema.py

import json
from pathlib import Path
from tests.helpers.solver_step2_output_schema import EXPECTED_STEP2_SCHEMA

# Load the final solver output schema from the JSON file
FINAL_SCHEMA_PATH = Path(__file__).parents[2] / "schema" / "solver_output_schema.json"

with FINAL_SCHEMA_PATH.open() as f:
    FINAL_SCHEMA = json.load(f)

# Extract the top-level keys from the final schema
FINAL_KEYS = set(FINAL_SCHEMA["properties"].keys())


def test_step2_schema_is_subset_of_final_schema():
    """
    Ensures Step 2 output is structurally compatible with the final solver output.
    """
    for key in EXPECTED_STEP2_SCHEMA:
        assert key in FINAL_KEYS, f"Missing in final schema: {key}"
