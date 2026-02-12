# tests/step1/test_step1_schema_output.py

import json
from pathlib import Path

from src.step1.orchestrate_step1 import orchestrate_step1


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    from jsonschema import validate
    validate(instance=instance, schema=schema)


def make_minimal_step1_input():
    """
    Minimal valid input for Step 1.
    """
    return {
        "domain": {
            "nx": 1,
            "ny": 1,
            "nz": 1,
            "x_length": 1.0,
            "y_length": 1.0,
            "z_length": 1.0,
        },
        "initial_conditions": {
            "initial_pressure": 0.0,
            "initial_velocity": [0.0, 0.0, 0.0],
        },
        "boundary_conditions": [],
        "physical_properties": {
            "rho": 1.0,
            "mu": 1.0,
        },
    }


def test_step1_output_matches_schema():
    """
    Contract test:
    Step‑1 output MUST validate against step1_output_schema.json.
    """

    schema_path = (
        Path(__file__).resolve().parents[2] / "schema" / "step1_output_schema.json"
    )
    schema = load_schema(str(schema_path))

    state_in = make_minimal_step1_input()

    state_out = orchestrate_step1(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    validate_json_schema(state_out, schema)
