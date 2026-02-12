# tests/step4/test_step4_apply_bc_velocity.py

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
# 4.3 Velocity Inlet on x_min (Staggered Consistency)
# ----------------------------------------------------------------------
def test_velocity_inlet_xmin():
    nx = ny = nz = 3
    u_in, v_in, w_in = 2.0, -1.0, 0.5

    bcs = [
        {"type": "inlet", "faces": ["x_min"], "velocity": [u_in, v_in, w_in]}
    ]

    state = _make_state(nx, ny, nz, bcs)
    out = apply_all_boundary_conditions(state)

    U_ext = out["U_ext"]
    V_ext = out["V_ext"]
    W_ext = out["W_ext"]

    # Normal component (U) at x_min must equal inlet value
    assert np.all(U_ext[0, :, :] == u_in)

    # Tangential components must be set so interpolation yields v_in, w_in
    # For now we only check they were modified from initialization
    assert not np.all(V_ext[0, :, :] == 0.0)
    assert not np.all(W_ext[0, :, :] == 0.0)


# ----------------------------------------------------------------------
# 4.4 Velocity Outlet on x_max
# ----------------------------------------------------------------------
def test_velocity_outlet_xmax():
    nx = ny = nz = 3
    bcs = [
        {"type": "outlet", "faces": ["x_max"]}
    ]

    state = _make_state(nx, ny, nz, bcs)

    # Fill interior with known pattern
    U_ext = state["U_ext"]
    for i in range(1, nx + 2):
        U_ext[i, :, :] = i * 1.0

    out = apply_all_boundary_conditions(state)
    U_ext = out["U_ext"]

    # Outlet: ghost = interior at x_max
    assert np.all(U_ext[-1, :, :] == U_ext[-2, :, :])


# ----------------------------------------------------------------------
# 4.5 No‑Slip Wall on y_min
# ----------------------------------------------------------------------
def test_no_slip_wall_ymin():
    nx = ny = nz = 3
    bcs = [
        {"type": "no-slip", "faces": ["y_min"]}
    ]

    state = _make_state(nx, ny, nz, bcs)

    # Give interior some non-zero velocities
    state["U_ext"][1:-2, 1, :] = 3.0
    state["V_ext"][:, 1, :] = -2.0
    state["W_ext"][:, 1, :] = 1.5

    out = apply_all_boundary_conditions(state)

    U_ext = out["U_ext"]
    V_ext = out["V_ext"]
    W_ext = out["W_ext"]

    # Wall velocities must be zero
    assert np.all(U_ext[:, 0, :] == 0.0)
    assert np.all(V_ext[:, 0, :] == 0.0)
    assert np.all(W_ext[:, 0, :] == 0.0)

    # Ghost normal/tangential must be negated interior
    assert np.all(U_ext[:, 0, :] == -U_ext[:, 1, :])
    assert np.all(V_ext[:, 0, :] == -V_ext[:, 1, :])
    assert np.all(W_ext[:, 0, :] == -W_ext[:, 1, :])


# ----------------------------------------------------------------------
# 4.6 Slip Wall on y_max
# ----------------------------------------------------------------------
def test_slip_wall_ymax():
    nx = ny = nz = 3
    bcs = [
        {"type": "slip", "faces": ["y_max"]}
    ]

    state = _make_state(nx, ny, nz, bcs)

    # Give interior some non-zero velocities
    state["U_ext"][:, -2, :] = 4.0
    state["V_ext"][:, -2, :] = -3.0
    state["W_ext"][:, -2, :] = 2.0

    out = apply_all_boundary_conditions(state)

    U_ext = out["U_ext"]
    V_ext = out["V_ext"]
    W_ext = out["W_ext"]

    # Normal component (V) must be zero at wall
    assert np.all(V_ext[:, -1, :] == 0.0)

    # Ghost normal = - interior normal
    assert np.all(V_ext[:, -1, :] == -V_ext[:, -2, :])

    # Tangential components mirrored
    assert np.all(U_ext[:, -1, :] == U_ext[:, -2, :])
    assert np.all(W_ext[:, -1, :] == W_ext[:, -2, :])


# ----------------------------------------------------------------------
# 4.7 Symmetry on z_min (Velocity + Pressure)
# ----------------------------------------------------------------------
def test_symmetry_zmin():
    nx = ny = nz = 3
    bcs = [
        {"type": "symmetry", "faces": ["z_min"]}
    ]

    state = _make_state(nx, ny, nz, bcs)

    # Give interior some non-zero velocities
    state["W_ext"][:, :, 1] = 5.0  # normal component
    state["U_ext"][:, :, 1] = 2.0  # tangential
    state["V_ext"][:, :, 1] = -1.0 # tangential

    out = apply_all_boundary_conditions(state)

    U_ext = out["U_ext"]
    V_ext = out["V_ext"]
    W_ext = out["W_ext"]
    P_ext = out["P_ext"]

    # Normal component zeroed
    assert np.all(W_ext[:, :, 0] == 0.0)

    # Tangential mirrored
    assert np.all(U_ext[:, :, 0] == U_ext[:, :, 1])
    assert np.all(V_ext[:, :, 0] == V_ext[:, :, 1])

    # Pressure Neumann: ghost = interior
    assert np.all(P_ext[:, :, 0] == P_ext[:, :, 1])
