# tests/step1/test_validate_physical_constraints.py

import pytest
import numpy as np

from src.step1.validate_physical_constraints import validate_physical_constraints
from src.step1.types import SolverState, GridConfig, Constants


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def make_state(**overrides):
    """
    Construct a minimal valid SolverState for Step 1 physical validation.
    Individual fields can be overridden for targeted tests.
    """
    grid = GridConfig(
        nx=2, ny=2, nz=2,
        dx=1.0, dy=1.0, dz=1.0,
        x_min=0.0, x_max=2.0,
        y_min=0.0, y_max=2.0,
        z_min=0.0, z_max=2.0,
    )

    fields = {
        "U": np.zeros((2, 2, 2)),
        "V": np.zeros((2, 2, 2)),
        "W": np.zeros((2, 2, 2)),
        "P": np.zeros((2, 2, 2)),
    }

    mask = np.ones((2, 2, 2), dtype=int)

    constants = Constants(
        rho=1.0,
        mu=0.1,
        dt=0.1,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        inv_dx=1.0,
        inv_dy=1.0,
        inv_dz=1.0,
    )

    state = SolverState(
        grid=grid,
        fields=fields,
        mask=mask,
        boundary_conditions={},
        constants=constants,
        config={},  # not used by physical constraints
    )

    # Apply overrides
    for key, value in overrides.items():
        setattr(state, key, value)

    return state


# ============================================================
# 1. FLUID PROPERTIES
# ============================================================

def test_density_must_be_positive():
    state = make_state()
    state.constants.rho = 0.0
    with pytest.raises(ValueError):
        validate_physical_constraints(state)


def test_viscosity_must_be_non_negative():
    state = make_state()
    state.constants.mu = -1.0
    with pytest.raises(ValueError):
        validate_physical_constraints(state)


# ============================================================
# 2. DOMAIN EXTENTS
# ============================================================

def test_domain_extents_must_be_ordered():
    state = make_state()
    state.grid.x_max = state.grid.x_min  # invalid
    with pytest.raises(ValueError):
        validate_physical_constraints(state)


def test_domain_extents_must_be_finite():
    state = make_state()
    state.grid.x_min = float("inf")
    with pytest.raises(ValueError):
        validate_physical_constraints(state)


# ============================================================
# 3. GRID COUNTS
# ============================================================

def test_grid_counts_must_be_positive():
    state = make_state()
    state.grid.nx = 0
    with pytest.raises(ValueError):
        validate_physical_constraints(state)


# ============================================================
# 4. MASK VALIDATION
# ============================================================

def test_mask_shape_must_match_grid():
    state = make_state()
    state.mask = np.ones((3, 2, 2), dtype=int)  # wrong shape
    with pytest.raises(ValueError):
        validate_physical_constraints(state)


def test_mask_entries_must_be_valid():
    state = make_state()
    state.mask[0, 0, 0] = 9  # invalid (must be -1, 0, or 1)
    with pytest.raises(ValueError):
        validate_physical_constraints(state)


# ============================================================
# 5. INITIAL CONDITIONS (IMPLICIT IN FIELD VALUES)
# ============================================================

def test_initial_velocity_fields_must_be_finite():
    state = make_state()
    state.fields["U"][0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        validate_physical_constraints(state)


def test_initial_pressure_fields_must_be_finite():
    state = make_state()
    state.fields["P"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        validate_physical_constraints(state)


# ============================================================
# 6. TIME STEP
# ============================================================

def test_time_step_must_be_positive():
    state = make_state()
    state.constants.dt = 0.0
    with pytest.raises(ValueError):
        validate_physical_constraints(state)
