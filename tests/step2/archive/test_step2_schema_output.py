# tests/step2/test_step2_schema_output.py

import json
from pathlib import Path

from src.step2.orchestrate_step2 import orchestrate_step2
from src.common.json_safe import to_json_safe
from tests.helpers.step1_schema_dummy_state import Step1SchemaDummyState


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    from jsonschema import validate
    validate(instance=instance, schema=schema)


def make_minimal_step2_input():
    """
    Minimal valid Step‑1 output that Step‑2 can accept.
    Use the official Step‑1 schema‑compliant dummy.
    """
    return Step1SchemaDummyState(nx=1, ny=1, nz=1)


def test_step2_output_matches_schema():
    """
    Contract test:
    Step‑2 output MUST validate against step2_output_schema.json.
    """

    schema_path = (
        Path(__file__).resolve().parents[2] / "schema" / "step2_output_schema.json"
    )
    schema = load_schema(str(schema_path))

    state_in = make_minimal_step2_input()

    state_out = orchestrate_step2(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    # Convert to JSON‑safe form BEFORE schema validation
    state_json = to_json_safe(state_out)

    validate_json_schema(state_json, schema)
