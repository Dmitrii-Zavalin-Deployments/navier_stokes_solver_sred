# tests/step4/test_step4_domain_block.py

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
    Uses a 2×2×2 grid so that coordinates, ghost layers,
    and index ranges are non‑degenerate.
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


def test_step4_domain_block_structure():
    """
    Contract test:
    Step‑4 MUST produce a complete, schema‑compliant 'domain' block.
    """

    state_in = make_minimal_step3_state()

    state_out = orchestrate_step4(
        state_in,
        validate_json_schema=validate_json_schema,
        load_schema=load_schema,
    )

    assert "domain" in state_out, "Step‑4 output must contain 'domain' block"
    domain = state_out["domain"]

    # ---------------------------------------------------------
    # Required top-level keys
    # ---------------------------------------------------------
    for key in [
        "coordinates",
        "ghost_layers",
        "index_ranges",
        "stencil_maps",
        "interpolation_maps",
    ]:
        assert key in domain, f"'domain' must contain '{key}'"

    # ---------------------------------------------------------
    # Coordinates
    # ---------------------------------------------------------
    coords = domain["coordinates"]
    for key in [
        "x_centers",
        "y_centers",
        "z_centers",
        "x_faces_u",
        "y_faces_v",
        "z_faces_w",
    ]:
        assert key in coords, f"'coordinates' must contain '{key}'"
        assert isinstance(coords[key], list), f"'{key}' must be a list"

    # Expected lengths for nx = ny = nz = 2
    assert len(coords["x_centers"]) == 2
    assert len(coords["y_centers"]) == 2
    assert len(coords["z_centers"]) == 2

    assert len(coords["x_faces_u"]) == 3
    assert len(coords["y_faces_v"]) == 3
    assert len(coords["z_faces_w"]) == 3

    # ---------------------------------------------------------
    # Ghost layers
    # ---------------------------------------------------------
    ghost = domain["ghost_layers"]
    for key in ["p_ext", "u_ext", "v_ext", "w_ext"]:
        assert key in ghost, f"'ghost_layers' must contain '{key}'"
        assert isinstance(ghost[key], list), f"'{key}' must be a list"
        assert len(ghost[key]) == 2, f"'{key}' must be [lo, hi]"

    # ---------------------------------------------------------
    # Index ranges
    # ---------------------------------------------------------
    idx = domain["index_ranges"]
    for key in [
        "interior",
        "ghost_x_lo",
        "ghost_x_hi",
        "ghost_y_lo",
        "ghost_y_hi",
        "ghost_z_lo",
        "ghost_z_hi",
    ]:
        assert key in idx, f"'index_ranges' must contain '{key}'"
        assert isinstance(idx[key], str), f"'{key}' must be a string"

    # ---------------------------------------------------------
    # Stencil maps
    # ---------------------------------------------------------
    stencils = domain["stencil_maps"]
    for key in ["xp", "xm", "yp", "ym", "zp", "zm"]:
        assert key in stencils, f"'stencil_maps' must contain '{key}'"
        assert isinstance(stencils[key], list), f"'{key}' must be a list"

    # ---------------------------------------------------------
    # Interpolation maps
    # ---------------------------------------------------------
    interp = domain["interpolation_maps"]
    for key in [
        "interp_u_to_v",
        "interp_u_to_w",
        "interp_v_to_u",
        "interp_v_to_w",
        "interp_w_to_u",
        "interp_w_to_v",
    ]:
        assert key in interp, f"'interpolation_maps' must contain '{key}'"
        assert isinstance(interp[key], dict), f"'{key}' must be a dict"
