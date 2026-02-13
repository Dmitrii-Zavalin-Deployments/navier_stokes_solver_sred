# tests/step1/test_step1_schema_output.py

import json
from pathlib import Path

from src.step1.orchestrate_step1 import orchestrate_step1
from src.common.json_safe import to_json_safe
from tests.helpers.minimal_step1_input import make_minimal_step1_input


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    from jsonschema import validate
    validate(instance=instance, schema=schema)


def test_step1_output_matches_schema():
    """
    Contract test:
    Step‑1 output MUST validate against step1_output_schema.json.
    """

    schema_path = (
        Path(__file__).resolve().parents[2] / "schema" / "step1_output_schema.json"
    )
    schema = load_schema(str(schema_path))

    # Use the shared minimal Step‑1 input
    state_in = make_minimal_step1_input()

    state_out = orchestrate_step1(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    # Convert JSON‑safe mirror again (for consistency with Step‑2/3/4 tests)
    state_json = to_json_safe(state_out["state_as_dict"])

    validate_json_schema(state_json, schema)
