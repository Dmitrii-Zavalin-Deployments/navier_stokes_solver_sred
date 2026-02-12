# tests/step4/test_step4_apply_bc_status_and_minimal_grid.py

import numpy as np
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState
from src.step4.allocate_extended_fields import allocate_extended_fields
from src.step4.initialize_staggered_fields import initialize_staggered_fields
from src.step4.apply_all_boundary_conditions import apply_all_boundary_conditions


def _make_state(nx, ny, nz, bcs):
    """
    Constructs a Step‑3‑compatible state and applies the first two Step‑4
    substeps (allocation + initialization), so the BC function can be tested
    in isolation.
    """
    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)
    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz
    state["config"]["boundary_conditions"] = bcs

    # Step 4.1: allocate extended fields
    state = allocate_extended_fields(state)

    # Step 4.2: initialize fields
    state["config"]["initial_conditions"] = {
        "initial_pressure": 1.0,
        "initial_velocity": [0.0, 0.0, 0.0],
    }
    state = initialize_staggered_fields(state)

    return state


# ----------------------------------------------------------------------
# 4.12 BCApplied Status Flags
# ----------------------------------------------------------------------
def test_bc_applied_status_flags():
    nx = ny = nz = 3
    bcs = [
        {"type": "inlet", "faces": ["x_min"], "velocity": [1.0, 0.0, 0.0]},
        {"type": "no-slip", "faces": ["y_min"]},
        {"type": "pressure_neumann", "faces": ["z_max"]},
    ]

    state = _make_state(nx, ny, nz, bcs)
    out = apply_all_boundary_conditions(state)

    status = out["BCApplied"]["boundary_conditions_status"]

    # All faces in BC table must be marked "applied"
    assert status["x_min"] == "applied"
    assert status["y_min"] == "applied"
    assert status["z_max"] == "applied"

    # Faces not mentioned must be "skipped"
    assert status["x_max"] == "skipped"
    assert status["y_max"] == "skipped"
    assert status["z_min"] == "skipped"


# ----------------------------------------------------------------------
# 4.14 Minimal Grid BC Application (nx=1, ny=1, nz=1)
# ----------------------------------------------------------------------
def test_minimal_grid_bc_application():
    nx = ny = nz = 1

    bcs = [
        {"type": "inlet", "faces": ["x_min"], "velocity": [2.0, 0.0, 0.0]},
        {"type": "no-slip", "faces": ["y_min"]},
        {"type": "symmetry", "faces": ["z_min"]},
    ]

    state = _make_state(nx, ny, nz, bcs)
    out = apply_all_boundary_conditions(state)

    U_ext = out["U_ext"]
    V_ext = out["V_ext"]
    W_ext = out["W_ext"]
    P_ext = out["P_ext"]

    # x_min inlet: U_ext[0,:,:] must equal inlet velocity
    assert np.all(U_ext[0, :, :] == 2.0)

    # y_min no-slip: velocities must be zero
    assert np.all(U_ext[:, 0, :] == 0.0)
    assert np.all(V_ext[:, 0, :] == 0.0)
    assert np.all(W_ext[:, 0, :] == 0.0)

    # z_min symmetry: normal component zero, tangential mirrored
    assert np.all(W_ext[:, :, 0] == 0.0)
    assert np.all(U_ext[:, :, 0] == U_ext[:, :, 1])
    assert np.all(V_ext[:, :, 0] == V_ext[:, :, 1])

    # Pressure Neumann for symmetry
    assert np.all(P_ext[:, :, 0] == P_ext[:, :, 1])
