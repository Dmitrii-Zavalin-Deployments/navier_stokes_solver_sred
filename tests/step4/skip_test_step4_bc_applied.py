# tests/step4/test_step4_bc_applied.py

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


def test_step4_bc_applied_structure():
    """
    Contract test:
    Step‑4 MUST produce a complete, schema‑compliant bc_applied block.

    Requirements:
    - all required flags exist
    - all 6 faces exist
    - each face has a valid structure
    """

    state_in = make_minimal_step3_state()

    state_out = orchestrate_step4(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    assert "bc_applied" in state_out, "Step‑4 output must contain 'bc_applied' block"
    bc = state_out["bc_applied"]

    # ---------------------------------------------------------
    # Required boolean flags
    # ---------------------------------------------------------
    for key in [
        "initial_velocity_enforced",
        "pressure_initial_applied",
        "velocity_initial_applied",
        "ghost_cells_filled",
        "boundary_cells_checked",
    ]:
        assert key in bc, f"'bc_applied' must contain '{key}'"
        assert isinstance(bc[key], bool), f"'{key}' must be a boolean"

    # ---------------------------------------------------------
    # Boundary faces
    # ---------------------------------------------------------
    assert "boundary_conditions_status" in bc, \
        "'bc_applied' must contain 'boundary_conditions_status'"

    faces = bc["boundary_conditions_status"]

    expected_faces = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]

    for face in expected_faces:
        assert face in faces, f"Missing face '{face}' in boundary_conditions_status"
        assert isinstance(faces[face], dict), f"Face '{face}' must be a dict"
        assert "applied" in faces[face], f"Face '{face}' must contain 'applied'"
        assert isinstance(faces[face]["applied"], bool), \
            f"'applied' for face '{face}' must be a boolean"

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------
    assert "version" in bc, "'bc_applied' must contain 'version'"
    assert "timestamp_applied" in bc, "'bc_applied' must contain 'timestamp_applied'"
    assert isinstance(bc["version"], str), "'version' must be a string"
    assert isinstance(bc["timestamp_applied"], str), "'timestamp_applied' must be a string"
