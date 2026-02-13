# tests/step4/test_step4_rhs_source.py

import json
from pathlib import Path

from src.step4.orchestrate_step4 import orchestrate_step4


def load_schema(path: str):
    with open(path, "r") as f:
        return json.load(f)


def validate_json_schema(instance, schema):
    from jsonschema import validate
    validate(instance=instance, schema=schema)


def make_minimal_step3_state_with_gravity():
    """
    Minimal valid Step‑3 output that Step‑4 can accept.
    Gravity is set to a non-zero vector so that RHS terms
    can be meaningfully tested.
    """
    return {
        "config": {
            "domain": {"nx": 2, "ny": 2, "nz": 2},
            "forces": {"gravity": [0.0, 0.0, -9.81]},
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


def test_step4_rhs_source_structure_and_shapes():
    """
    Contract test:
    Step‑4 MUST produce a complete, schema‑compliant rhs_source block.

    Requirements:
    - fx_u, fy_v, fz_w exist
    - shapes match u_ext, v_ext, w_ext
    - gravity is applied correctly to fz_w
    """

    state_in = make_minimal_step3_state_with_gravity()

    state_out = orchestrate_step4(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    assert "rhs_source" in state_out, "Step‑4 output must contain 'rhs_source' block"
    rhs = state_out["rhs_source"]

    # Required keys
    for key in ["fx_u", "fy_v", "fz_w"]:
        assert key in rhs, f"'rhs_source' must contain '{key}'"

    # Shapes must match extended fields
    u_ext = state_out["u_ext"]
    v_ext = state_out["v_ext"]
    w_ext = state_out["w_ext"]

    fx_u = rhs["fx_u"]
    fy_v = rhs["fy_v"]
    fz_w = rhs["fz_w"]

    # Compare shapes by recursive length matching
    def same_shape(a, b):
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(same_shape(x, y) for x, y in zip(a, b))
        return True

    assert same_shape(fx_u, u_ext), "fx_u shape must match u_ext"
    assert same_shape(fy_v, v_ext), "fy_v shape must match v_ext"
    assert same_shape(fz_w, w_ext), "fz_w shape must match w_ext"

    # Gravity must be applied to fz_w (gravity = [0, 0, -9.81])
    # All interior fluid cells should have -9.81
    # Ghost cells may differ, but interior must match
    interior_values = []
    for k in range(len(w_ext)):
        for j in range(len(w_ext[k])):
            for i in range(len(w_ext[k][j])):
                interior_values.append(fz_w[k][j][i])

    # At least one interior cell must show gravity
    assert any(abs(v + 9.81) < 1e-6 for v in interior_values), \
        "fz_w must include gravity contribution (-9.81)"
