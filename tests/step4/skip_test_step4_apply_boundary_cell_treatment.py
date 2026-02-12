# tests/step4/test_step4_apply_boundary_cell_treatment.py

import numpy as np
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState
from src.step4.allocate_extended_fields import allocate_extended_fields
from src.step4.initialize_staggered_fields import initialize_staggered_fields
from src.step4.apply_all_boundary_conditions import apply_all_boundary_conditions
from src.step4.apply_boundary_cell_treatment import apply_boundary_cell_treatment


def _prepare_state(nx, ny, nz, bcs, mask=None):
    """
    Shared setup for all boundary‑cell treatment tests:
      1. Step‑3 dummy state
      2. allocate_extended_fields
      3. initialize_staggered_fields
      4. apply_all_boundary_conditions
    """
    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)

    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz
    state["config"]["boundary_conditions"] = bcs

    # Optional mask override
    if mask is not None:
        state["mask"] = mask
        state["is_fluid"] = (mask == 1)

    # Step 4.1
    state = allocate_extended_fields(state)

    # Step 4.2
    state["config"]["initial_conditions"] = {
        "initial_pressure": 1.0,
        "initial_velocity": [0.0, 0.0, 0.0],
    }
    state = initialize_staggered_fields(state)

    # Step 4.3
    state = apply_all_boundary_conditions(state)

    return state


# ----------------------------------------------------------------------
# 5.1 No‑Slip Boundary‑Fluid Damping
# ----------------------------------------------------------------------
def test_boundary_fluid_no_slip_damping():
    nx = ny = nz = 3

    # Boundary-fluid cell at (1,1,1)
    mask = np.ones((nx, ny, nz), dtype=int)
    mask[1, 1, 1] = -1  # boundary-fluid

    bcs = [{"type": "no-slip", "faces": ["x_min"]}]

    state = _prepare_state(nx, ny, nz, bcs, mask)
    out = apply_boundary_cell_treatment(state)

    # Boundary-fluid cell velocities must be damped toward zero
    U_ext = out["U_ext"]
    V_ext = out["V_ext"]
    W_ext = out["W_ext"]

    assert abs(U_ext[2, 2, 2]) <= abs(state["U_ext"][2, 2, 2])
    assert abs(V_ext[1, 2, 2]) <= abs(state["V_ext"][1, 2, 2])
    assert abs(W_ext[1, 1, 2]) <= abs(state["W_ext"][1, 1, 2])


# ----------------------------------------------------------------------
# 5.2 Slip Boundary‑Fluid Behavior
# ----------------------------------------------------------------------
def test_boundary_fluid_slip_behavior():
    nx = ny = nz = 3

    mask = np.ones((nx, ny, nz), dtype=int)
    mask[1, 1, 1] = -1  # boundary-fluid

    bcs = [{"type": "slip", "faces": ["y_min"]}]

    state = _prepare_state(nx, ny, nz, bcs, mask)
    out = apply_boundary_cell_treatment(state)

    # Normal component should be damped, tangential preserved
    U_ext = out["U_ext"]
    V_ext = out["V_ext"]
    W_ext = out["W_ext"]

    # Normal to y_min is V
    assert abs(V_ext[1, 1, 1]) <= abs(state["V_ext"][1, 1, 1])

    # Tangential components should remain unchanged
    assert U_ext[2, 1, 1] == state["U_ext"][2, 1, 1]
    assert W_ext[1, 1, 2] == state["W_ext"][1, 1, 2]


# ----------------------------------------------------------------------
# 5.3 Inlet/Outlet/Symmetry Boundary‑Fluid
# ----------------------------------------------------------------------
def test_boundary_fluid_inlet_outlet_symmetry():
    nx = ny = nz = 3

    mask = np.ones((nx, ny, nz), dtype=int)
    mask[1, 1, 1] = -1  # boundary-fluid

    bcs = [
        {"type": "inlet", "faces": ["x_min"], "velocity": [3.0, 0.0, 0.0]},
        {"type": "outlet", "faces": ["y_min"]},
        {"type": "symmetry", "faces": ["z_min"]},
    ]

    state = _prepare_state(nx, ny, nz, bcs, mask)
    out = apply_boundary_cell_treatment(state)

    U_ext = out["U_ext"]
    V_ext = out["V_ext"]
    W_ext = out["W_ext"]

    # Boundary-fluid cell should inherit inlet velocity if aligned
    assert U_ext[2, 2, 2] == 3.0

    # Outlet: normal component copied
    assert V_ext[1, 1, 1] == state["V_ext"][1, 1, 1]

    # Symmetry: normal zeroed
    assert W_ext[1, 1, 1] == 0.0


# ----------------------------------------------------------------------
# 5.4 BCApplied.boundary_cells_checked
# ----------------------------------------------------------------------
def test_boundary_cells_checked_flag():
    nx = ny = nz = 3

    mask = np.ones((nx, ny, nz), dtype=int)
    mask[0, 0, 0] = -1  # boundary-fluid

    bcs = [{"type": "no-slip", "faces": ["x_min"]}]

    state = _prepare_state(nx, ny, nz, bcs, mask)
    out = apply_boundary_cell_treatment(state)

    assert out["BCApplied"].get("boundary_cells_checked") is True


# ----------------------------------------------------------------------
# 5.5 Minimal Grid with Boundary‑Fluid
# ----------------------------------------------------------------------
def test_boundary_fluid_minimal_grid():
    nx = ny = nz = 1

    mask = np.array([[[ -1 ]]], dtype=int)  # single boundary-fluid cell

    bcs = [{"type": "inlet", "faces": ["x_min"], "velocity": [1.0, 0.0, 0.0]}]

    state = _prepare_state(nx, ny, nz, bcs, mask)
    out = apply_boundary_cell_treatment(state)

    U_ext = out["U_ext"]
    V_ext = out["V_ext"]
    W_ext = out["W_ext"]

    # Boundary-fluid cell should inherit inlet velocity
    assert U_ext[1, 1, 1] == 1.0

    # No NaNs or infs
    assert np.isfinite(U_ext).all()
    assert np.isfinite(V_ext).all()
    assert np.isfinite(W_ext).all()
