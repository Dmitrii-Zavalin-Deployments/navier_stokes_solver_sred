# tests/step4/test_step4_final_flags.py

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


def test_step4_final_flags():
    """
    Contract test:
    Step‑4 MUST set final flags:
        initialized == True
        ready_for_time_loop == True
    """

    state_in = make_minimal_step3_state()

    state_out = orchestrate_step4(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    # Flags must exist
    assert "initialized" in state_out, "'initialized' flag missing"
    assert "ready_for_time_loop" in state_out, "'ready_for_time_loop' flag missing"

    # Flags must be True
    assert state_out["initialized"] is True, "'initialized' must be True"
    assert state_out["ready_for_time_loop"] is True, "'ready_for_time_loop' must be True"
