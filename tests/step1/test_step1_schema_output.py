# tests/step1/test_step1_schema_output.py

import json
from pathlib import Path
import numpy as np

from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.minimal_step1_input import make_minimal_step1_input


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    from jsonschema import validate
    validate(instance=instance, schema=schema)


# ------------------------------------------------------------
# JSON‑safe conversion helper (same pattern as Step‑2 and Step‑3)
# ------------------------------------------------------------
def to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(x) for x in obj]
    return obj


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

    # Convert JSON‑safe mirror again (for consistency with Step‑2/3 tests)
    state_json = to_json_safe(state_out["state_as_dict"])

    validate_json_schema(state_json, schema)
