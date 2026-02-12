# tests/step4/test_step4_allocate_extended_fields.py

import numpy as np
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState
from src.step4.allocate_extended_fields import allocate_extended_fields


def _make_step3_state(nx, ny, nz):
    """
    Helper to construct a Step‑3‑compatible state for Step‑4 allocation tests.
    """
    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)
    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz
    return state


# ----------------------------------------------------------------------
# 2.1 Shape and Ghost Size (3D)
# ----------------------------------------------------------------------
def test_allocate_extended_fields_shape_and_ghost_size():
    nx, ny, nz = 4, 3, 2
    state = _make_step3_state(nx, ny, nz)

    out = allocate_extended_fields(state)

    assert np.asarray(out["P_ext"]).shape == (nx + 2, ny + 2, nz + 2)
    assert np.asarray(out["U_ext"]).shape == (nx + 3, ny + 2, nz + 2)
    assert np.asarray(out["V_ext"]).shape == (nx, ny + 3, nz + 2)
    assert np.asarray(out["W_ext"]).shape == (nx, ny, nz + 3)


# ----------------------------------------------------------------------
# 2.2 Interior Copy Correctness
# ----------------------------------------------------------------------
def test_allocate_extended_fields_interior_copy_correctness():
    nx, ny, nz = 3, 3, 3
    state = _make_step3_state(nx, ny, nz)

    # Fill fields with a known pattern
    P = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                P[i, j, k] = i + j + k

    state["fields"]["P"] = P
    state["fields"]["U"] = np.ones((nx + 1, ny, nz))
    state["fields"]["V"] = np.ones((nx, ny + 1, nz)) * 2
    state["fields"]["W"] = np.ones((nx, ny, nz + 1)) * 3

    out = allocate_extended_fields(state)

    # Interior slices must match original fields
    assert np.all(out["P_ext"][1:-1, 1:-1, 1:-1] == P)
    assert np.all(out["U_ext"][1:-2, 1:-1, 1:-1] == state["fields"]["U"])
    assert np.all(out["V_ext"][0:nx, 1:-2, 1:-1] == state["fields"]["V"])
    assert np.all(out["W_ext"][0:nx, 0:ny, 1:-2] == state["fields"]["W"])


# ----------------------------------------------------------------------
# 2.3 Ghost Layers Zeroed
# ----------------------------------------------------------------------
def test_allocate_extended_fields_ghost_layers_zeroed():
    nx, ny, nz = 3, 3, 3
    state = _make_step3_state(nx, ny, nz)

    out = allocate_extended_fields(state)

    P_ext = np.asarray(out["P_ext"])
    U_ext = np.asarray(out["U_ext"])
    V_ext = np.asarray(out["V_ext"])
    W_ext = np.asarray(out["W_ext"])

    # Check that ghost regions are zero
    assert np.all(P_ext[0, :, :] == 0)
    assert np.all(P_ext[-1, :, :] == 0)
    assert np.all(P_ext[:, 0, :] == 0)
    assert np.all(P_ext[:, -1, :] == 0)
    assert np.all(P_ext[:, :, 0] == 0)
    assert np.all(P_ext[:, :, -1] == 0)

    # Same for U, V, W
    assert np.all(U_ext[0, :, :] == 0)
    assert np.all(U_ext[-1, :, :] == 0)
    assert np.all(V_ext[:, 0, :] == 0)
    assert np.all(V_ext[:, -1, :] == 0)
    assert np.all(W_ext[:, :, 0] == 0)
    assert np.all(W_ext[:, :, -1] == 0)


# ----------------------------------------------------------------------
# 2.4 IndexRanges and GhostLayers Views
# ----------------------------------------------------------------------
def test_allocate_extended_fields_indexranges_and_views():
    nx, ny, nz = 3, 3, 3
    state = _make_step3_state(nx, ny, nz)

    out = allocate_extended_fields(state)
    domain = out["Domain"]

    # Modify a ghost slice through the view
    ghost_slice = domain["GhostLayers"]["P_ext"]["GHOST_X_LO"]
    out["P_ext"][ghost_slice] = 123.0

    # Underlying array must reflect the change
    assert np.all(out["P_ext"][0, :, :] == 123.0)


# ----------------------------------------------------------------------
# 2.5 Dtype and NaN Robustness
# ----------------------------------------------------------------------
def test_allocate_extended_fields_dtype_and_nan_robustness():
    nx, ny, nz = 3, 3, 3
    state = _make_step3_state(nx, ny, nz)

    # Insert NaNs and float32
    P = np.ones((nx, ny, nz), dtype=np.float32)
    P[1, 1, 1] = np.nan
    state["fields"]["P"] = P

    out = allocate_extended_fields(state)
    P_ext = np.asarray(out["P_ext"])

    # Ghosts must be zero (no NaNs)
    assert not np.isnan(P_ext[0]).any()
    assert not np.isnan(P_ext[-1]).any()

    # Interior NaN must be preserved
    assert np.isnan(P_ext[2, 2, 2])


# ----------------------------------------------------------------------
# 2.6 Minimal Grid Allocation (nx=1, ny=1, nz=1)
# ----------------------------------------------------------------------
def test_allocate_extended_fields_minimal_grid():
    nx = ny = nz = 1
    state = _make_step3_state(nx, ny, nz)

    out = allocate_extended_fields(state)

    assert np.asarray(out["P_ext"]).shape == (3, 3, 3)
    assert np.asarray(out["U_ext"]).shape == (4, 3, 3)
    assert np.asarray(out["V_ext"]).shape == (1, 4, 3)
    assert np.asarray(out["W_ext"]).shape == (1, 1, 4)

    # No NaNs or infs
    assert np.isfinite(out["P_ext"]).all()
    assert np.isfinite(out["U_ext"]).all()
    assert np.isfinite(out["V_ext"]).all()
    assert np.isfinite(out["W_ext"]).all()
