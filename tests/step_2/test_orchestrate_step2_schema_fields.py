# tests/step_2/test_orchestrate_step2_schema_fields.py

import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2
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

    return {
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


def test_orchestrate_step2_schema_fields():
    dummy = DummyState(4, 4, 4)
    state = _make_state_from_dummy(dummy)

    result = orchestrate_step2(state)

    required = [
        "constants",
        "fields",
        "is_fluid",
        "is_boundary_cell",
        "operators",
        "ppe",
        "health",
        "advection_meta",
    ]

    for key in required:
        assert key in result, f"Missing required field: {key}"
