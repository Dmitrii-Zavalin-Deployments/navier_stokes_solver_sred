# tests/schema/test_step4_schema_vs_final_schema.py

from tests.helpers.solver_step4_output_schema import step4_output_schema
from tests.helpers.final_output_schema import final_output_schema


def test_step4_schema_is_subset_of_final_schema():
    """
    Step 4 output must be valid final output.
    """
    for key in step4_output_schema["required"]:
        assert key in final_output_schema["required"], f"{key} missing in final schema"
