# tests/step4/test_step4_precompute_rhs_source_terms.py

import numpy as np
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState
from src.step4.allocate_extended_fields import allocate_extended_fields
from src.step4.initialize_staggered_fields import initialize_staggered_fields
from src.step4.apply_all_boundary_conditions import apply_all_boundary_conditions
from src.step4.apply_boundary_cell_treatment import apply_boundary_cell_treatment
from src.step4.precompute_rhs_source_terms import precompute_rhs_source_terms


def _prepare_state(nx, ny, nz, gravity):
    """
    Shared setup for RHS source term tests:
      1. Step‑3 dummy state
      2. allocate_extended_fields
      3. initialize_staggered_fields
      4. apply_all_boundary_conditions (no BCs needed here)
      5. apply_boundary_cell_treatment
    """
    state = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)

    state["config"]["domain"]["nx"] = nx
    state["config"]["domain"]["ny"] = ny
    state["config"]["domain"]["nz"] = nz

    # Gravity vector
    state["config"]["forces"] = {"gravity": gravity}

    # Step 4.1
    state = allocate_extended_fields(state)

    # Step 4.2
    state["config"]["initial_conditions"] = {
        "initial_pressure": 0.0,
        "initial_velocity": [0.0, 0.0, 0.0],
    }
    state = initialize_staggered_fields(state)

    # Step 4.3 (no BCs)
    state["config"]["boundary_conditions"] = []
    state = apply_all_boundary_conditions(state)

    # Step 4.4
    state = apply_boundary_cell_treatment(state)

    return state


# ----------------------------------------------------------------------
# 6.1 Constant Gravity Field
# ----------------------------------------------------------------------
def test_rhs_constant_gravity_field():
    nx = ny = nz = 3
    gravity = [0.0, 0.0, -9.81]

    state = _prepare_state(nx, ny, nz, gravity)
    out = precompute_rhs_source_terms(state)

    RHS = out["RHS_Source"]

    # All interior cells must have identical RHS contribution
    interior = RHS["P"][1:-1, 1:-1, 1:-1]
    assert np.allclose(interior, interior[0, 0, 0])


# ----------------------------------------------------------------------
# 6.2 Non‑Zero All Components
# ----------------------------------------------------------------------
def test_rhs_nonzero_all_components():
    nx = ny = nz = 3
    gravity = [1.0, -2.0, 3.0]

    state = _prepare_state(nx, ny, nz, gravity)
    out = precompute_rhs_source_terms(state)

    RHS = out["RHS_Source"]

    # Pressure RHS should reflect divergence of gravity field
    interior = RHS["P"][1:-1, 1:-1, 1:-1]
    assert np.isfinite(interior).all()
    assert not np.all(interior == 0.0)


# ----------------------------------------------------------------------
# 6.3 Use of Face Coordinates
# ----------------------------------------------------------------------
def test_rhs_use_of_face_coordinates():
    nx = ny = nz = 3
    gravity = [0.0, 0.0, -1.0]

    state = _prepare_state(nx, ny, nz, gravity)

    # Inject synthetic face coordinates
    coords = np.zeros((nx + 2, ny + 2, nz + 2, 3))
    for i in range(nx + 2):
        for j in range(ny + 2):
            for k in range(nz + 2):
                coords[i, j, k] = [i, j, k]

    state["Domain"]["FaceCoords"] = coords

    out = precompute_rhs_source_terms(state)
    RHS = out["RHS_Source"]["P"]

    # RHS must vary with k (z-coordinate)
    assert not np.all(RHS[1:-1, 1:-1, 1:-1] == RHS[1, 1, 1])


# ----------------------------------------------------------------------
# 6.4 Zero External Force
# ----------------------------------------------------------------------
def test_rhs_zero_external_force():
    nx = ny = nz = 3
    gravity = [0.0, 0.0, 0.0]

    state = _prepare_state(nx, ny, nz, gravity)
    out = precompute_rhs_source_terms(state)

    RHS = out["RHS_Source"]["P"]

    # Entire RHS must be zero
    assert np.all(RHS == 0.0)


# ----------------------------------------------------------------------
# 6.5 Minimal Grid RHS (nx=1, ny=1, nz=1)
# ----------------------------------------------------------------------
def test_rhs_minimal_grid():
    nx = ny = nz = 1
    gravity = [0.0, 0.0, -9.81]

    state = _prepare_state(nx, ny, nz, gravity)
    out = precompute_rhs_source_terms(state)

    RHS = out["RHS_Source"]["P"]

    # Shape must be valid
    assert RHS.shape == (nx + 2, ny + 2, nz + 2)

    # No NaNs or infs
    assert np.isfinite(RHS).all()
