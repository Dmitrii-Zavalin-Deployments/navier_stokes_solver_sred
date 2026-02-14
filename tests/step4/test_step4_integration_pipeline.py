# tests/step4/test_step4_integration_pipeline.py

import numpy as np
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState
from src.step4.initialize_staggered_fields import initialize_staggered_fields


def _make_state(nx, ny, nz):
    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)
    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz
    return state


# ----------------------------------------------------------------------
# 1. Full pipeline: allocation + initialization + mask semantics
# ----------------------------------------------------------------------
def test_step4_pipeline_basic_connectivity():
    nx = ny = nz = 3
    state = _make_state(nx, ny, nz)

    state["config"]["initial_conditions"] = {
        "initial_pressure": 5.0,
        "initial_velocity": [1.0, 2.0, 3.0],
    }

    out = initialize_staggered_fields(state)

    # Extended fields must exist
    assert "P_ext" in out
    assert "U_ext" in out
    assert "V_ext" in out
    assert "W_ext" in out

    # Shapes must match staggered expectations
    assert out["P_ext"].shape == (nx + 2, ny + 2, nz + 2)
    assert out["U_ext"].shape == (nx + 3, ny + 2, nz + 2)
    assert out["V_ext"].shape == (nx, ny + 3, nz + 2)
    assert out["W_ext"].shape == (nx, ny, nz + 3)

    # Interior values must match initial conditions
    assert np.all(out["P_ext"][1:-1, 1:-1, 1:-1] == 5.0)
    assert np.all(out["U_ext"][1:-1, 1:-1, 1:-1] == 1.0)
    assert np.all(out["V_ext"][:, 1:-1, 1:-1] == 2.0)
    assert np.all(out["W_ext"][:, :, 1:-1] == 3.0)


# ----------------------------------------------------------------------
# 2. Mask semantics: solid → zero, boundary-fluid → preserved
# ----------------------------------------------------------------------
def test_step4_pipeline_mask_semantics():
    nx = ny = nz = 3
    state = _make_state(nx, ny, nz)

    mask = np.ones((nx, ny, nz), dtype=int)
    mask[1, 1, 1] = 0      # solid
    mask[0, 0, 0] = -1     # boundary-fluid

    state["mask"] = mask
    state["is_fluid"] = (mask == 1)

    state["config"]["initial_conditions"] = {
        "initial_pressure": 10.0,
        "initial_velocity": [4.0, 5.0, 6.0],
    }

    out = initialize_staggered_fields(state)

    # Solid cell must be zero
    assert out["P_ext"][2, 2, 2] == 0.0

    # Boundary-fluid cell must be preserved
    assert out["P_ext"][1, 1, 1] == 10.0


# ----------------------------------------------------------------------
# 3. BC vs mask conflict: solid mask wins
# ----------------------------------------------------------------------
def test_step4_pipeline_bc_vs_mask_conflict():
    nx = ny = nz = 3
    state = _make_state(nx, ny, nz)

    mask = np.ones((nx, ny, nz), dtype=int)
    mask[0, :, :] = 0  # solid boundary
    state["mask"] = mask
    state["is_fluid"] = (mask == 1)

    state["config"]["boundary_conditions"] = [
        {"type": "inlet", "faces": ["x_min"], "velocity": [9.0, 0.0, 0.0]}
    ]

    state["config"]["initial_conditions"] = {
        "initial_pressure": 1.0,
        "initial_velocity": [9.0, 0.0, 0.0],
    }

    out = initialize_staggered_fields(state)

    # Solid mask must override BCs → velocities must be zero
    assert np.all(out["U_ext"][0, :, :] == 0.0)


# ----------------------------------------------------------------------
# 4. Domain block connectivity
# ----------------------------------------------------------------------
def test_step4_pipeline_domain_block_consistency():
    nx = ny = nz = 3
    state = _make_state(nx, ny, nz)

    state["config"]["initial_conditions"] = {
        "initial_pressure": 2.0,
        "initial_velocity": [1.0, 1.0, 1.0],
    }

    out = initialize_staggered_fields(state)

    domain = out["Domain"]

    # Domain must expose extended arrays
    assert domain["P_ext"] is out["P_ext"]
    assert domain["U_ext"] is out["U_ext"]
    assert domain["V_ext"] is out["V_ext"]
    assert domain["W_ext"] is out["W_ext"]

    # GhostLayers must exist and be index tuples
    gl = domain["GhostLayers"]["P_ext"]
    assert isinstance(gl["GHOST_X_LO"], tuple)
    assert isinstance(gl["GHOST_X_HI"], tuple)


# ----------------------------------------------------------------------
# 5. Minimal grid connectivity
# ----------------------------------------------------------------------
def test_step4_pipeline_minimal_grid():
    nx = ny = nz = 1
    state = _make_state(nx, ny, nz)

    state["config"]["initial_conditions"] = {
        "initial_pressure": 3.0,
        "initial_velocity": [0.5, 0.5, 0.5],
    }

    out = initialize_staggered_fields(state)

    assert out["P_ext"].shape == (3, 3, 3)
    assert out["U_ext"].shape == (4, 3, 3)
    assert out["V_ext"].shape == (1, 4, 3)
    assert out["W_ext"].shape == (1, 1, 4)

    # Interior values must be correct
    assert np.all(out["P_ext"][1:-1, 1:-1, 1:-1] == 3.0)
    assert np.all(out["U_ext"][1:-1, 1:-1, 1:-1] == 0.5)
