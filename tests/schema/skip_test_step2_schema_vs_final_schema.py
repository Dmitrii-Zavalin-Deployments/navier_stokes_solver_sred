# tests/schema/test_step2_schema_vs_final_schema.py

from tests.helpers.solver_step2_output_schema import EXPECTED_STEP2_SCHEMA
from tests.helpers.solver_final_output_schema import EXPECTED_FINAL_SCHEMA


def test_step2_schema_is_subset_of_final_schema():
    """
    Ensures Step 2 output is structurally compatible with the final solver output.
    """
    for key in EXPECTED_STEP2_SCHEMA:
        assert key in EXPECTED_FINAL_SCHEMA, f"Missing in final schema: {key}"
