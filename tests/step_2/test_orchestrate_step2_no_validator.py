# tests/step_2/test_orchestrate_step2_no_validator.py

import numpy as np
import src.step2.orchestrate_step2 as o
from tests.helpers.dummy_state_step2 import DummyState


def _make_state_from_dummy(dummy: DummyState) -> dict:
    """
    Convert DummyState (old-style) into a Step‑1‑schema‑compliant dict.
    """
    grid = dummy["Grid"]
    cfg = dummy["Config"]
    mask = dummy["Mask"]
    const = dummy["Constants"]
    boundary_table = dummy.get("BoundaryTable", [])

    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    dx, dy, dz = grid["dx"], grid["dy"], grid["dz"]

    state = {
        "config": {
            "boundary_conditions": cfg.get("boundary_conditions", {}),
            "domain": cfg.get("domain", {}),
            "fluid": cfg.get("fluid_properties", {}),
            "forces": cfg.get("forces", {}),
            "geometry_definition": cfg.get("geometry_definition", {}),
            "simulation": cfg.get("simulation_parameters", {}),
        },
        "grid": {
            "x_min": 0.0,
            "x_max": dx * nx,
            "y_min": 0.0,
            "y_max": dy * ny,
            "z_min": 0.0,
            "z_max": dz * nz,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": dx,
            "dy": dy,
            "dz": dz,
        },
        "fields": {
            "P": np.zeros((nx, ny, nz), float),
            "U": dummy.U,
            "V": dummy.V,
            "W": dummy.W,
            "Mask": mask,
        },
        "mask_3d": mask,
        "boundary_table": boundary_table,
        "constants": const,
    }

    return state


def test_orchestrate_step2_no_validator():
    # Simulate missing validator
    o.validate_json_schema = None

    dummy = DummyState(4, 4, 4)
    state = _make_state_from_dummy(dummy)

    result = o.orchestrate_step2(state)

    # NEW: Step 2 stores constants under lowercase "constants"
    assert "constants" in result
