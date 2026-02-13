# tests/step4/test_step4_field_renaming.py

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


def test_step4_field_renaming():
    """
    Step‑4 must rename extended fields:
        P_ext → p_ext
        U_ext → u_ext
        V_ext → v_ext
        W_ext → w_ext

    And old names must not remain in the final state.
    """

    state_in = make_minimal_step3_state()

    state_out = orchestrate_step4(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    # New names must exist
    assert "p_ext" in state_out, "Expected 'p_ext' after renaming"
    assert "u_ext" in state_out, "Expected 'u_ext' after renaming"
    assert "v_ext" in state_out, "Expected 'v_ext' after renaming"
    assert "w_ext" in state_out, "Expected 'w_ext' after renaming"

    # Old names must NOT exist
    assert "P_ext" not in state_out, "Old name 'P_ext' must not remain"
    assert "U_ext" not in state_out, "Old name 'U_ext' must not remain"
    assert "V_ext" not in state_out, "Old name 'V_ext' must not remain"
    assert "W_ext" not in state_out, "Old name 'W_ext' must not remain"
