# tests/step4/test_step4_conservation_symmetry.py

import numpy as np
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState

from src.step4.allocate_extended_fields import allocate_extended_fields
from src.step4.initialize_staggered_fields import initialize_staggered_fields
from src.step4.apply_all_boundary_conditions import apply_all_boundary_conditions
from src.step4.apply_boundary_cell_treatment import apply_boundary_cell_treatment
from src.step4.precompute_rhs_source_terms import precompute_rhs_source_terms
from src.step4.verify_post_bc_state import verify_post_bc_state


def _run_full_pipeline(state):
    """
    Runs the entire Step‑4 pipeline on a prepared Step‑3 dummy state.
    """
    state = allocate_extended_fields(state)
    state = initialize_staggered_fields(state)
    state = apply_all_boundary_conditions(state)
    state = apply_boundary_cell_treatment(state)
    state = precompute_rhs_source_terms(state)
    state = verify_post_bc_state(state)
    return state


# ----------------------------------------------------------------------
# 10.1 Global Mass Balance (Initial Closed Box)
# ----------------------------------------------------------------------
def test_global_mass_balance_closed_box():
    nx = ny = nz = 4

    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)

    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz

    # Closed box: symmetry on all faces
    state["config"]["boundary_conditions"] = [
        {"type": "symmetry", "faces": ["x_min"]},
        {"type": "symmetry", "faces": ["x_max"]},
        {"type": "symmetry", "faces": ["y_min"]},
        {"type": "symmetry", "faces": ["y_max"]},
        {"type": "symmetry", "faces": ["z_min"]},
        {"type": "symmetry", "faces": ["z_max"]},
    ]

    state["config"]["initial_conditions"] = {
        "initial_pressure": 1.0,
        "initial_velocity": [0.0, 0.0, 0.0],
    }

    state["config"]["forces"] = {"gravity": [0.0, 0.0, 0.0]}

    out = _run_full_pipeline(state)

    # Divergence must remain zero in a closed box
    assert out["Verification"]["divergence_ok"] is True
    assert out["Verification"]["max_divergence"] == 0.0


# ----------------------------------------------------------------------
# 10.2 Mirror Symmetry Preservation
# ----------------------------------------------------------------------
def test_mirror_symmetry_preservation():
    nx = ny = nz = 4

    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)

    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz

    # Symmetric initial velocity field
    U = np.zeros((nx + 2, ny + 2, nz + 2))
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            for k in range(1, nz + 1):
                U[i, j, k] = abs(i - (nx // 2 + 1))

    state["U_ext"] = U.copy()

    state["config"]["initial_conditions"] = {
        "initial_pressure": 1.0,
        "initial_velocity": [0.0, 0.0, 0.0],
    }

    # Symmetry on all faces
    state["config"]["boundary_conditions"] = [
        {"type": "symmetry", "faces": ["x_min"]},
        {"type": "symmetry", "faces": ["x_max"]},
        {"type": "symmetry", "faces": ["y_min"]},
        {"type": "symmetry", "faces": ["y_max"]},
        {"type": "symmetry", "faces": ["z_min"]},
        {"type": "symmetry", "faces": ["z_max"]},
    ]

    state["config"]["forces"] = {"gravity": [0.0, 0.0, 0.0]}

    out = _run_full_pipeline(state)

    U_final = out["U_ext"]

    # Mirror symmetry must be preserved across x-midplane
    for j in range(1, ny + 1):
        for k in range(1, nz + 1):
            assert np.allclose(
                U_final[1:nx+1, j, k],
                U_final[nx:0:-1, j, k]
            )
