# tests/step4/test_step4_schema_output.py

import json
from pathlib import Path

import pytest

from src.step4.orchestrate_step4 import orchestrate_step4


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    """
    We intentionally do NOT import jsonschema here.
    The real project uses a wrapper function for validation.
    This stub allows the test to run until the real validator is wired in.
    """
    from jsonschema import validate
    validate(instance=instance, schema=schema)


def make_minimal_step3_output():
    """
    Construct the smallest valid Step‑3 output that Step‑4 can accept.
    This will evolve as Step‑4 grows.
    """
    return {
        "P": [[[0.0]]],
        "U": [[[0.0]]],
        "V": [[[0.0]]],
        "W": [[[0.0]]],
        "mask": [[[1]]],  # single fluid cell
        "config": {
            "domain": {
                "nx": 1,
                "ny": 1,
                "nz": 1,
            },
            "forces": {
                "gravity": [0.0, 0.0, 0.0]
            },
            "initial_conditions": {
                "initial_velocity": [0.0, 0.0, 0.0]
            },
            "boundary_conditions": []
        }
    }


def test_step4_output_matches_schema():
    """
    Contract test:
    Step‑4 output MUST validate against step4_output_schema.json.
    """

    # ---------------------------------------------------------
    # Load schema
    # ---------------------------------------------------------
    schema_path = (
        Path(__file__).resolve().parents[2] / "schema" / "step4_output_schema.json"
    )
    schema = load_schema(str(schema_path))

    # ---------------------------------------------------------
    # Prepare minimal Step‑3 output
    # ---------------------------------------------------------
    state_in = make_minimal_step3_output()

    # ---------------------------------------------------------
    # Run Step‑4
    # ---------------------------------------------------------
    state_out = orchestrate_step4(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    # ---------------------------------------------------------
    # Validate Step‑4 output
    # ---------------------------------------------------------
    validate_json_schema(state_out, schema)
