# tests/schema/test_step4_schema_vs_final_schema.py

import json
from pathlib import Path
from tests.helpers.solver_step4_output_schema import step4_output_schema

# Load the final solver output schema from the JSON file
FINAL_SCHEMA_PATH = Path(__file__).parents[2] / "schema" / "solver_output_schema.json"

with FINAL_SCHEMA_PATH.open() as f:
    FINAL_SCHEMA = json.load(f)

# Extract required keys from the final schema
FINAL_REQUIRED = set(FINAL_SCHEMA["required"])


def test_step4_schema_is_subset_of_final_schema():
    """
    Step 4 output must be structurally compatible with the final solver output.
    """
    for key in step4_output_schema["required"]:
        assert key in FINAL_REQUIRED, f"{key} missing in final schema"
