# tests/schema/test_step2_schema_vs_final_schema.py

import json
from tests.helpers.solver_step2_output_schema import solver_step2_output_schema


def test_step2_schema_is_subset_of_final_schema():
    with open("solver_output_schema.json", "r") as f:
        final_schema = json.load(f)

    step2_schema = solver_step2_output_schema

    # ------------------------------------------------------------
    # Recursive subset check
    # ------------------------------------------------------------
    def is_subset(sub, sup):
        if isinstance(sub, dict):
            assert isinstance(sup, dict)
            for key, val in sub.items():
                assert key in sup
                is_subset(val, sup[key])
        else:
            # leaf nodes: type or placeholder
            assert True

    is_subset(step2_schema, final_schema)
