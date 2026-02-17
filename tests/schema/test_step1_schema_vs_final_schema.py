# tests/schema/test_step1_schema_vs_final_schema.py

import json
from pathlib import Path

from tests.helpers.solver_step1_output_schema import STEP1_OUTPUT_SCHEMA


def load_final_schema():
    schema_path = Path(__file__).resolve().parents[2] / "schema" / "solver_output_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_step1_schema_is_subset_of_final_schema():
    final_schema = load_final_schema()

    # Top-level keys
    for key in STEP1_OUTPUT_SCHEMA.keys():
        assert key in final_schema["properties"], (
            f"Step 1 field '{key}' missing in final solver_output_schema.json"
        )

    # Grid
    for key in STEP1_OUTPUT_SCHEMA["grid"].keys():
        assert key in final_schema["properties"]["grid"]["properties"], (
            f"grid.{key} missing in final schema"
        )

    # Config
    for key in STEP1_OUTPUT_SCHEMA["config"].keys():
        assert key in final_schema["properties"]["config"]["properties"], (
            f"config.{key} missing in final schema"
        )

    # Constants
    for key in STEP1_OUTPUT_SCHEMA["constants"].keys():
        assert key in final_schema["properties"]["constants"]["properties"], (
            f"constants.{key} missing in final schema"
        )

    # Fields
    for key in STEP1_OUTPUT_SCHEMA["fields"].keys():
        assert key in final_schema["properties"]["fields"]["properties"], (
            f"fields.{key} missing in final schema"
        )
