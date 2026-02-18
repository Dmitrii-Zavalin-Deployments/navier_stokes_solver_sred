# tests/schema/test_step5_schema_vs_final_schema.py

import json
from pathlib import Path
import jsonschema

from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy
from tests.helpers.solver_step5_output_schema import step5_output_schema

# Load the final solver output schema from the JSON file
FINAL_SCHEMA_PATH = Path(__file__).parents[2] / "schema" / "solver_output_schema.json"

with FINAL_SCHEMA_PATH.open() as f:
    FINAL_SCHEMA = json.load(f)


def test_step5_schema_is_compatible_with_final_schema():
    """
    Ensures that a valid Step 5 output is also valid
    under the final output schema.
    """
    dummy = make_step5_output_dummy()

    # Validate against Step 5 schema
    jsonschema.validate(instance=dummy, schema=step5_output_schema)

    # Validate against final output schema
    jsonschema.validate(instance=dummy, schema=FINAL_SCHEMA)
