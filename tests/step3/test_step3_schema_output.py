# tests/step3/test_step3_schema_output.py

import json
from pathlib import Path

from src.step3.orchestrate_step3 import orchestrate_step3


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    from jsonschema import validate
    validate(instance=instance, schema=schema)


def make_minimal_step3_input():
    """
    Minimal valid Step‑2 output that Step‑3 can accept.
    """
    return {
        "U": [[[0.0]]],
        "V": [[[0.0]]],
        "W": [[[0.0]]],
        "P": [[[0.0]]],
        "mask": [[[1]]],
        "operators": {},
        "config": {
            "domain": {"nx": 1, "ny": 1, "nz": 1},
            "forces": {"gravity": [0.0, 0.0, 0.0]},
            "physical_properties": {"rho": 1.0, "mu": 1.0},
            "boundary_conditions": [],
        },
    }


def test_step3_output_matches_schema():
    """
    Contract test:
    Step‑3 output MUST validate against step3_output_schema.json.
    """

    schema_path = (
        Path(__file__).resolve().parents[2] / "schema" / "step3_output_schema.json"
    )
    schema = load_schema(str(schema_path))

    state_in = make_minimal_step3_input()

    state_out = orchestrate_step3(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    validate_json_schema(state_out, schema)
