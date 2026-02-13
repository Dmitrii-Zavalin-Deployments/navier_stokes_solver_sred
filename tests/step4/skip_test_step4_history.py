# tests/step4/test_step4_history.py

import json
from pathlib import Path

from src.step4.orchestrate_step4 import orchestrate_step4


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    from jsonschema import validate
    validate(instance=instance, schema=schema)


def make_minimal_step3_state():
    """
    Minimal valid Step‑3 output that Step‑4 can accept.
    """
    return {
        "config": {
            "domain": {"nx": 2, "ny": 2, "nz": 2},
            "forces": {"gravity": [0.0, 0.0, 0.0]},
            "initial_conditions": {"initial_velocity": [0.0, 0.0, 0.0]},
            "boundary_conditions": [],
        },
        "mask": [[[1, 1], [1, 1]]],
        "is_fluid": [[[True, True], [True, True]]],
        "is_boundary_cell": [[[False, False], [False, False]]],
        "fields": {
            "P": [[[0.0, 0.0], [0.0, 0.0]]],
            "U": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            "V": [[[0.0, 0.0], [0.0, 0.0]]],
            "W": [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
        },
        "bcs": [],
        "constants": {},
        "operators": {},
        "ppe": {"ppe_is_singular": False},
        "health": {
            "post_correction_divergence_norm": 0.0,
            "max_velocity_magnitude": 0.0,
            "cfl_advection_estimate": 0.0,
        },
        "advection_meta": None,
        "history": {},
    }


def test_step4_history_block_empty():
    """
    Contract test:
    Step‑4 MUST produce a 'history' block with empty arrays.

    Requirements:
    - history exists
    - times, divergence_norms, max_velocity_history, ppe_iterations_history are empty
    """

    state_in = make_minimal_step3_state()

    state_out = orchestrate_step4(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    assert "history" in state_out, "Step‑4 output must contain 'history' block"
    hist = state_out["history"]

    # Required keys
    for key in [
        "times",
        "divergence_norms",
        "max_velocity_history",
        "ppe_iterations_history",
    ]:
        assert key in hist, f"'history' must contain '{key}'"
        assert isinstance(hist[key], list), f"'{key}' must be a list"
        assert len(hist[key]) == 0, f"'{key}' must be an empty list in Step‑4"
