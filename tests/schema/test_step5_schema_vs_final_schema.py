# tests/schema/test_step5_schema_vs_final_schema.py

import json
from pathlib import Path
from tests.helpers.solver_step5_output_schema import step5_output_schema

# Load the final solver output schema from the JSON file
FINAL_SCHEMA_PATH = Path(__file__).parents[2] / "schema" / "solver_output_schema.json"

with FINAL_SCHEMA_PATH.open() as f:
    FINAL_SCHEMA = json.load(f)

FINAL_REQUIRED = set(FINAL_SCHEMA["required"])


def test_step5_schema_is_subset_of_final_schema():
    """
    Step 5 output must be structurally compatible with the final solver output.
    Only required fields must be a subset of the final schema's required fields.
    """
    for key in step5_output_schema["required"]:
        assert key in FINAL_REQUIRED, f"{key} missing in final schema"
