# tests/step4/test_step4_schema_output.py

import json
from pathlib import Path
import numpy as np

import pytest

from src.step4.orchestrate_step4 import orchestrate_step4
from src.common.json_safe import to_json_safe
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    """
    Stub validator used in tests.
    The real project uses a wrapper function.
    """
    from jsonschema import validate
    validate(instance=instance, schema=schema)


def make_minimal_step3_output():
    """
    Minimal valid Step‑3 output that Step‑4 can accept.
    Use the official schema‑compliant dummy.
    """
    return Step3SchemaDummyState(nx=1, ny=1, nz=1)


def test_step4_output_matches_schema():
    """
    Contract test:
    Step‑4 output MUST validate against step4_output_schema.json.
    """

    schema_path = (
        Path(__file__).resolve().parents[2] / "schema" / "step4_output_schema.json"
    )
    schema = load_schema(str(schema_path))

    # Minimal Step‑3 output
    state_in = make_minimal_step3_output()

    # Run Step‑4
    state_out = orchestrate_step4(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    # Convert to JSON‑safe form BEFORE schema validation
    state_json = to_json_safe(state_out)

    validate_json_schema(state_json, schema)
