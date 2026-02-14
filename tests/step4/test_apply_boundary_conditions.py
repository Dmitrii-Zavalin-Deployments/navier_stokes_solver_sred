# tests/step4/test_apply_boundary_conditions.py

import numpy as np
from src.step4.apply_boundary_conditions import apply_boundary_conditions


def make_state(nx=2, ny=2, nz=2):
    return {
        "config": {
            "domain": {"nx": nx, "ny": ny, "nz": nz},
            "boundary_conditions": [],
        },
        "mask": np.ones((nx, ny, nz), dtype=int),
        "P_ext": np.zeros((nx+2, ny+2, nz+2)),
        "U_ext": np.zeros((nx+3, ny+2, nz+2)),
        "V_ext": np.zeros((nx+2, ny+3, nz+2)),
        "W_ext": np.zeros((nx+2, ny+2, nz+3)),
    }


def test_no_slip_xmin():
    state = make_state()
    state["config"]["boundary_conditions"] = [
        {"type": "no-slip", "variable": "u", "direction": "x", "side": "min"}
    ]

    state["U_ext"][1, :, :] = 5.0
    out = apply_boundary_conditions(state)

    assert np.allclose(out["U_ext"][0, :, :], -5.0)


def test_inlet_ymax():
    state = make_state()
    state["config"]["boundary_conditions"] = [
        {"type": "inlet", "variable": "v", "direction": "y", "side": "max", "value": 3.0}
    ]

    out = apply_boundary_conditions(state)
    assert np.allclose(out["V_ext"][:, -1, :], 3.0)


def test_pressure_neumann_zmin():
    state = make_state()
    state["P_ext"][:, :, 1] = 9.0
    state["config"]["boundary_conditions"] = [
        {"type": "neumann", "variable": "p", "direction": "z", "side": "min"}
    ]

    out = apply_boundary_conditions(state)
    assert np.allclose(out["P_ext"][:, :, 0], 9.0)
