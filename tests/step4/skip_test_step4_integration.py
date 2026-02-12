# tests/step4/test_step4_integration.py

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
# 9.1 Full Box with Mixed BCs
# ----------------------------------------------------------------------
def test_integration_full_box_mixed_bcs():
    nx = ny = nz = 4

    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)

    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz

    state["config"]["initial_conditions"] = {
        "initial_pressure": 1.0,
        "initial_velocity": [0.0, 0.0, 0.0],
    }

    state["config"]["boundary_conditions"] = [
        {"type": "inlet", "faces": ["x_min"], "velocity": [1.0, 0.0, 0.0]},
        {"type": "outlet", "faces": ["x_max"]},
        {"type": "no-slip", "faces": ["y_min"]},
        {"type": "slip", "faces": ["y_max"]},
        {"type": "symmetry", "faces": ["z_min"]},
        {"type": "pressure_neumann", "faces": ["z_max"]},
    ]

    state["config"]["forces"] = {"gravity": [0.0, 0.0, -9.81]}

    out = _run_full_pipeline(state)

    # Basic sanity checks
    assert "Verification" in out
    assert np.isfinite(out["Verification"]["max_divergence"])
    assert out["BCApplied"]["boundary_conditions_status"]["x_min"] == "applied"


# ----------------------------------------------------------------------
# 9.2 Complex Mask with Solids and Boundary‑Fluid
# ----------------------------------------------------------------------
def test_integration_complex_mask_boundary_fluid():
    nx = ny = nz = 4

    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)

    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz

    # Create a complex mask:
    # - solids in a diagonal
    # - boundary-fluid around them
    mask = np.ones((nx, ny, nz), dtype=int)
    for i in range(nx):
        mask[i, i % ny, i % nz] = 0  # solid
        if i + 1 < nx:
            mask[i + 1, i % ny, i % nz] = -1  # boundary-fluid

    state["mask"] = mask
    state["is_fluid"] = (mask == 1)

    state["config"]["initial_conditions"] = {
        "initial_pressure": 0.0,
        "initial_velocity": [0.0, 0.0, 0.0],
    }

    state["config"]["boundary_conditions"] = [
        {"type": "no-slip", "faces": ["x_min"]},
        {"type": "symmetry", "faces": ["z_min"]},
    ]

    state["config"]["forces"] = {"gravity": [0.0, -1.0, 0.0]}

    out = _run_full_pipeline(state)

    # Boundary-fluid cells must be checked
    assert out["BCApplied"]["boundary_cells_checked"] is True

    # No NaNs in final fields
    assert np.isfinite(out["U_ext"]).all()
    assert np.isfinite(out["V_ext"]).all()
    assert np.isfinite(out["W_ext"]).all()


# ----------------------------------------------------------------------
# 9.3 Regression: Known Stable Configuration
# ----------------------------------------------------------------------
def test_regression_known_stable_configuration():
    nx = ny = nz = 3

    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)

    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz

    # Known stable configuration:
    # - symmetry on all faces
    # - zero gravity
    # - zero initial velocity
    state["config"]["initial_conditions"] = {
        "initial_pressure": 1.0,
        "initial_velocity": [0.0, 0.0, 0.0],
    }

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

    # Regression expectation: divergence must remain zero
    assert out["Verification"]["divergence_ok"] is True
    assert out["Verification"]["max_divergence"] == 0.0
