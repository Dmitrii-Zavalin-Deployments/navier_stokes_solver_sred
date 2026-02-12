# tests/step4/test_step4_apply_bc_pressure.py

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
# 4.1 Pressure Dirichlet on Single Face
# ----------------------------------------------------------------------
def test_pressure_dirichlet_single_face():
    nx = ny = nz = 3
    bcs = [
        {"type": "pressure_dirichlet", "faces": ["x_min"], "value": 10.0}
    ]

    state = _make_state(nx, ny, nz, bcs)
    out = apply_all_boundary_conditions(state)

    P_ext = out["P_ext"]

    # Ghost slice at x_min must be exactly 10.0
    assert np.all(P_ext[0, :, :] == 10.0)

    # Interior slice must remain unchanged (initialized to 1.0)
    assert np.all(P_ext[1, :, :] == 1.0)


# ----------------------------------------------------------------------
# 4.2 Pressure Neumann on Single Face
# ----------------------------------------------------------------------
def test_pressure_neumann_single_face():
    nx = ny = nz = 3
    bcs = [
        {"type": "pressure_neumann", "faces": ["x_max"]}
    ]

    state = _make_state(nx, ny, nz, bcs)

    # Fill interior with a known pattern
    P_ext = state["P_ext"]
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            for k in range(1, nz + 1):
                P_ext[i, j, k] = i + j + k

    out = apply_all_boundary_conditions(state)
    P_ext = out["P_ext"]

    # Neumann: ghost = interior at x_max
    assert np.all(P_ext[-1, :, :] == P_ext[-2, :, :])


# ----------------------------------------------------------------------
# 4.11 Mixed Pressure BCs at Corner (Dirichlet > Neumann)
# ----------------------------------------------------------------------
def test_pressure_mixed_corner_dirichlet_over_neumann():
    nx = ny = nz = 3
    bcs = [
        {"type": "pressure_dirichlet", "faces": ["x_min"], "value": 5.0},
        {"type": "pressure_neumann", "faces": ["y_min"]},
    ]

    state = _make_state(nx, ny, nz, bcs)
    out = apply_all_boundary_conditions(state)

    P_ext = out["P_ext"]

    # Corner at (x_min, y_min) = (0, 0, :)
    # Dirichlet must win over Neumann
    assert np.all(P_ext[0, 0, :] == 5.0)
