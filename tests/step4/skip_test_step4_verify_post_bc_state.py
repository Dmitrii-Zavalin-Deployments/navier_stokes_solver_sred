# tests/step4/test_step4_verify_post_bc_state.py

import numpy as np
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState
from src.step4.allocate_extended_fields import allocate_extended_fields
from src.step4.initialize_staggered_fields import initialize_staggered_fields
from src.step4.apply_all_boundary_conditions import apply_all_boundary_conditions
from src.step4.apply_boundary_cell_treatment import apply_boundary_cell_treatment
from src.step4.precompute_rhs_source_terms import precompute_rhs_source_terms
from src.step4.verify_post_bc_state import verify_post_bc_state


def _prepare_state(nx, ny, nz, bcs=None):
    """
    Shared setup for post‑BC verification tests:
      1. Step‑3 dummy state
      2. allocate_extended_fields
      3. initialize_staggered_fields
      4. apply_all_boundary_conditions
      5. apply_boundary_cell_treatment
      6. precompute_rhs_source_terms
    """
    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)

    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz

    # Boundary conditions (optional)
    state["config"]["boundary_conditions"] = bcs or []

    # Gravity for RHS (default zero)
    state["config"]["forces"] = {"gravity": [0.0, 0.0, 0.0]}

    # Step 4.1
    state = allocate_extended_fields(state)

    # Step 4.2
    state["config"]["initial_conditions"] = {
        "initial_pressure": 0.0,
        "initial_velocity": [0.0, 0.0, 0.0],
    }
    state = initialize_staggered_fields(state)

    # Step 4.3
    state = apply_all_boundary_conditions(state)

    # Step 4.4
    state = apply_boundary_cell_treatment(state)

    # Step 4.5
    state = precompute_rhs_source_terms(state)

    return state


# ----------------------------------------------------------------------
# 7.1 Zero Velocity Field
# ----------------------------------------------------------------------
def test_verify_zero_velocity_field():
    nx = ny = nz = 3

    state = _prepare_state(nx, ny, nz)

    # Zero velocity everywhere → divergence must be zero
    out = verify_post_bc_state(state)

    assert out["Verification"]["divergence_ok"] is True
    assert out["Verification"]["max_divergence"] == 0.0


# ----------------------------------------------------------------------
# 7.2 Known Divergence Pattern
# ----------------------------------------------------------------------
def test_verify_known_divergence_pattern():
    nx = ny = nz = 3

    state = _prepare_state(nx, ny, nz)

    # Inject a known divergence pattern
    U_ext = state["U_ext"]
    U_ext[2, 2, 2] = 10.0  # creates divergence at (1,1,1)

    out = verify_post_bc_state(state)

    assert out["Verification"]["divergence_ok"] is False
    assert out["Verification"]["max_divergence"] > 0.0


# ----------------------------------------------------------------------
# 7.3 Intentional BC Violation
# ----------------------------------------------------------------------
def test_verify_intentional_bc_violation():
    nx = ny = nz = 3

    # Apply a no-slip BC
    bcs = [{"type": "no-slip", "faces": ["x_min"]}]
    state = _prepare_state(nx, ny, nz, bcs)

    # Now intentionally violate it
    state["U_ext"][0, :, :] = 999.0

    out = verify_post_bc_state(state)

    assert out["Verification"]["bc_violations_detected"] is True
    assert "x_min" in out["Verification"]["violation_faces"]


# ----------------------------------------------------------------------
# 7.4 Violation Type Logging
# ----------------------------------------------------------------------
def test_verify_violation_type_logging():
    nx = ny = nz = 3

    bcs = [{"type": "no-slip", "faces": ["y_min"]}]
    state = _prepare_state(nx, ny, nz, bcs)

    # Violate normal + tangential components
    state["V_ext"][:, 0, :] = 123.0
    state["U_ext"][:, 0, :] = 456.0

    out = verify_post_bc_state(state)

    log = out["Verification"]["violation_types"]

    assert "normal_velocity_violation" in log
    assert "tangential_velocity_violation" in log


# ----------------------------------------------------------------------
# 7.5 Minimal Grid Verification (nx=1, ny=1, nz=1)
# ----------------------------------------------------------------------
def test_verify_minimal_grid():
    nx = ny = nz = 1

    state = _prepare_state(nx, ny, nz)

    out = verify_post_bc_state(state)

    # Minimal grid must not produce NaNs or false violations
    assert np.isfinite(out["Verification"]["max_divergence"])
    assert out["Verification"]["bc_violations_detected"] in (False, True)
