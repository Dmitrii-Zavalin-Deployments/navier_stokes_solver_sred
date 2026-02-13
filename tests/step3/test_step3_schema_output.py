# tests/step3/test_step3_schema_output.py

import json
from pathlib import Path
import numpy as np

from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    from jsonschema import validate
    validate(instance=instance, schema=schema)


# ------------------------------------------------------------
# JSON‑safe conversion helper (mirrors Step‑1 and Step‑2 tests)
# ------------------------------------------------------------
def to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(x) for x in obj]
    return obj


def make_minimal_step3_input():
    """
    Minimal valid Step‑2 output that Step‑3 can accept.
    Use the official schema‑compliant dummy.
    """
    return Step3SchemaDummyState(nx=1, ny=1, nz=1)


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
        current_time=0.0,
        step_index=0,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    # Convert to JSON‑safe form BEFORE schema validation
    state_json = to_json_safe(state_out)

    validate_json_schema(state_json, schema)
