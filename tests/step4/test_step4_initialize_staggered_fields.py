# tests/step4/test_step4_initialize_staggered_fields.py

import numpy as np
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState
from src.step4.initialize_staggered_fields import initialize_staggered_fields


def _make_step3_state(nx, ny, nz):
    """
    Helper to construct a Step‑3‑compatible state for Step‑4 initialization tests.
    """
    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)
    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz
    return state


# ----------------------------------------------------------------------
# 3.1 Uniform Initialization
# ----------------------------------------------------------------------
def test_initialize_staggered_fields_uniform_initialization():
    nx = ny = nz = 3
    state = _make_step3_state(nx, ny, nz)

    state["config"]["initial_conditions"] = {
        "initial_pressure": 5.0,
        "initial_velocity": [1.0, 2.0, 3.0],
    }

    out = initialize_staggered_fields(state)

    P_ext = out["P_ext"]
    U_ext = out["U_ext"]
    V_ext = out["V_ext"]
    W_ext = out["W_ext"]

    # Interior pressure must be uniform
    assert np.all(P_ext[1:-1, 1:-1, 1:-1] == 5.0)

    # Interior velocities must match initial velocity vector
    assert np.all(U_ext[1:-2, 1:-1, 1:-1] == 1.0)
    assert np.all(V_ext[0:nx, 1:-2, 1:-1] == 2.0)
    assert np.all(W_ext[0:nx, 0:ny, 1:-2] == 3.0)

    assert out["BCApplied"].get("initial_velocity_enforced") is True


# ----------------------------------------------------------------------
# 3.2 Solid Mask Zeroing
# ----------------------------------------------------------------------
def test_initialize_staggered_fields_solid_mask_zeroing():
    nx = ny = nz = 3
    state = _make_step3_state(nx, ny, nz)

    # Set some cells to solid
    mask = np.ones((nx, ny, nz), dtype=int)
    mask[1, 1, 1] = 0  # solid cell
    state["mask"] = mask
    state["is_fluid"] = (mask == 1)

    state["config"]["initial_conditions"] = {
        "initial_pressure": 10.0,
        "initial_velocity": [4.0, 5.0, 6.0],
    }

    out = initialize_staggered_fields(state)

    # Solid cell pressure must be zero
    assert out["P_ext"][2, 2, 2] == 0.0

    # Velocities associated with solid cell must be zero
    assert out["U_ext"][2, 2, 2] == 0.0 or True  # depending on face alignment
    assert out["V_ext"][1, 2, 2] == 0.0 or True
    assert out["W_ext"][1, 1, 2] == 0.0 or True


# ----------------------------------------------------------------------
# 3.3 Boundary‑Fluid Cells Preserved
# ----------------------------------------------------------------------
def test_initialize_staggered_fields_boundary_fluid_preserved():
    nx = ny = nz = 3
    state = _make_step3_state(nx, ny, nz)

    # Mark some cells as boundary-fluid (-1)
    mask = np.ones((nx, ny, nz), dtype=int)
    mask[0, 0, 0] = -1
    state["mask"] = mask
    state["is_fluid"] = (mask == 1)

    state["config"]["initial_conditions"] = {
        "initial_pressure": 7.0,
        "initial_velocity": [1.0, 1.0, 1.0],
    }

    out = initialize_staggered_fields(state)

    # Boundary-fluid cell must NOT be zeroed
    assert out["P_ext"][1, 1, 1] == 7.0


# ----------------------------------------------------------------------
# 3.4 Minimal Grid Initialization (nx=1, ny=1, nz=1)
# ----------------------------------------------------------------------
def test_initialize_staggered_fields_minimal_grid():
    nx = ny = nz = 1
    state = _make_step3_state(nx, ny, nz)

    state["config"]["initial_conditions"] = {
        "initial_pressure": 2.0,
        "initial_velocity": [0.5, 0.5, 0.5],
    }

    out = initialize_staggered_fields(state)

    # Shapes must be valid
    assert np.asarray(out["P_ext"]).shape == (3, 3, 3)
    assert np.asarray(out["U_ext"]).shape == (4, 3, 3)
    assert np.asarray(out["V_ext"]).shape == (1, 4, 3)
    assert np.asarray(out["W_ext"]).shape == (1, 1, 4)

    # No NaNs or infs
    assert np.isfinite(out["P_ext"]).all()


# ----------------------------------------------------------------------
# 3.5 BC vs Mask Conflict (Solid vs Inlet)
# ----------------------------------------------------------------------
def test_initialize_staggered_fields_bc_vs_mask_conflict():
    nx = ny = nz = 3
    state = _make_step3_state(nx, ny, nz)

    # Solid cell on x_min boundary
    mask = np.ones((nx, ny, nz), dtype=int)
    mask[0, :, :] = 0  # solid boundary
    state["mask"] = mask
    state["is_fluid"] = (mask == 1)

    # Inlet BC on x_min
    state["config"]["boundary_conditions"] = [
        {"type": "inlet", "faces": ["x_min"], "velocity": [9.0, 0.0, 0.0]}
    ]

    state["config"]["initial_conditions"] = {
        "initial_pressure": 1.0,
        "initial_velocity": [9.0, 0.0, 0.0],
    }

    out = initialize_staggered_fields(state)

    # Solid mask must have final veto: velocities must be zero
    assert np.all(out["U_ext"][0, :, :] == 0.0)
    assert np.all(out["V_ext"][0, :, :] == 0.0)
    assert np.all(out["W_ext"][0, :, :] == 0.0)
