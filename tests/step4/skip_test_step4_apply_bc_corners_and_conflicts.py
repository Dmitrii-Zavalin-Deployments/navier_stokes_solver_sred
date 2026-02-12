# tests/step4/test_step4_apply_bc_corners_and_conflicts.py

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
# 4.8 Multiple BCs on Different Faces
# ----------------------------------------------------------------------
def test_multiple_bcs_on_different_faces():
    nx = ny = nz = 3
    bcs = [
        {"type": "inlet", "faces": ["x_min"], "velocity": [1.0, 0.0, 0.0]},
        {"type": "outlet", "faces": ["x_max"]},
        {"type": "no-slip", "faces": ["y_min"]},
        {"type": "slip", "faces": ["y_max"]},
        {"type": "symmetry", "faces": ["z_min"]},
        {"type": "pressure_neumann", "faces": ["z_max"]},
    ]

    state = _make_state(nx, ny, nz, bcs)
    out = apply_all_boundary_conditions(state)

    # Check that each face was touched (not necessarily physics-validated here)
    status = out["BCApplied"]["boundary_conditions_status"]

    assert status["x_min"] == "applied"
    assert status["x_max"] == "applied"
    assert status["y_min"] == "applied"
    assert status["y_max"] == "applied"
    assert status["z_min"] == "applied"
    assert status["z_max"] == "applied"


# ----------------------------------------------------------------------
# 4.9 Corner Conflict: No‑Slip vs Inlet
# ----------------------------------------------------------------------
def test_corner_conflict_no_slip_vs_inlet():
    nx = ny = nz = 3
    bcs = [
        {"type": "no-slip", "faces": ["x_min"]},
        {"type": "inlet", "faces": ["y_min"], "velocity": [5.0, 0.0, 0.0]},
    ]

    state = _make_state(nx, ny, nz, bcs)
    out = apply_all_boundary_conditions(state)

    U_ext = out["U_ext"]
    V_ext = out["V_ext"]
    W_ext = out["W_ext"]

    # Corner at (x_min, y_min) = (0, 0, :)
    # Priority: No-slip > Inlet
    # So velocities must be zero at the corner
    assert np.all(U_ext[0, 0, :] == 0.0)
    assert np.all(V_ext[0, 0, :] == 0.0)
    assert np.all(W_ext[0, 0, :] == 0.0)


# ----------------------------------------------------------------------
# 4.10 Corner Conflict: Inlet vs Outlet vs Symmetry
# ----------------------------------------------------------------------
def test_corner_conflict_inlet_vs_outlet_vs_symmetry():
    nx = ny = nz = 3
    bcs = [
        {"type": "inlet", "faces": ["x_min"], "velocity": [9.0, 0.0, 0.0]},
        {"type": "outlet", "faces": ["y_min"]},
        {"type": "symmetry", "faces": ["z_min"]},
    ]

    state = _make_state(nx, ny, nz, bcs)
    out = apply_all_boundary_conditions(state)

    U_ext = out["U_ext"]

    # Corner at (x_min, y_min, z_min) = (0, 0, 0)
    # Priority: Inlet > Outlet > Symmetry
    # So U_ext[0,0,0] must equal inlet velocity
    assert U_ext[0, 0, 0] == 9.0


# ----------------------------------------------------------------------
# 4.13 Ghost Consistency After Mixed BCs
# ----------------------------------------------------------------------
def test_ghost_consistency_after_mixed_bcs():
    nx = ny = nz = 3
    bcs = [
        {"type": "inlet", "faces": ["x_min"], "velocity": [1.0, 0.0, 0.0]},
        {"type": "no-slip", "faces": ["y_min"]},
    ]

    state = _make_state(nx, ny, nz, bcs)
    out = apply_all_boundary_conditions(state)

    U_ext = out["U_ext"]
    V_ext = out["V_ext"]

    # x_min face should follow inlet
    assert np.all(U_ext[0, :, :] == 1.0)

    # y_min face should follow no-slip
    assert np.all(V_ext[:, 0, :] == 0.0)

    # Shared edge (x_min, y_min) must follow priority: no-slip > inlet
    assert np.all(U_ext[0, 0, :] == 0.0)
    assert np.all(V_ext[0, 0, :] == 0.0)
