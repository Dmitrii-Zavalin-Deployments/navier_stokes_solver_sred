# tests/step1/test_validate_physical_constraints_math.py

import pytest
import numpy as np

from src.step1.validate_physical_constraints import validate_physical_constraints
from src.step1.orchestrate_step1 import orchestrate_step1_state
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

# ---------------------------------------------------------
# Helper: Use the dummy to create a valid state object
# ---------------------------------------------------------

def get_valid_state():
    """
    Returns a valid SolverState object by passing the canonical 
    dummy through the orchestrator.
    """
    json_input = solver_input_schema_dummy()
    return orchestrate_step1_state(json_input)


# ============================================================
# 1. FLUID PROPERTIES
# ============================================================

def test_density_must_be_positive():
    state = get_valid_state()
    state.constants["rho"] = 0.0  # Attribute access
    with pytest.raises(ValueError, match="density"):
        validate_physical_constraints(state)


def test_viscosity_must_be_non_negative():
    state = get_valid_state()
    state.constants["mu"] = -1.0
    with pytest.raises(ValueError, match="viscosity"):
        validate_physical_constraints(state)


# ============================================================
# 2. DOMAIN EXTENTS
# ============================================================

def test_domain_extents_must_be_ordered():
    state = get_valid_state()
    # x_max must be > x_min. state.grid is a dict stored in the object.
    state.grid["x_max"] = state.grid["x_min"]
    with pytest.raises(ValueError, match="extent"):
        validate_physical_constraints(state)


def test_domain_extents_must_be_finite():
    state = get_valid_state()
    state.grid["x_min"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_physical_constraints(state)


# ============================================================
# 3. GRID COUNTS
# ============================================================

def test_grid_counts_must_be_positive():
    state = get_valid_state()
    state.grid["nx"] = 0
    with pytest.raises(ValueError, match="nx"):
        validate_physical_constraints(state)


# ============================================================
# 4. MASK VALIDATION
# ============================================================

def test_mask_shape_must_match_grid():
    state = get_valid_state()
    # The dummy is 2x2x2; we provide a mismatched 3x2x2 directly to the attribute
    state.mask = np.ones((3, 2, 2), dtype=int)
    with pytest.raises(ValueError, match="shape"):
        validate_physical_constraints(state)


def test_mask_entries_must_be_valid():
    state = get_valid_state()
    state.mask[0, 0, 0] = 9  # Only -1, 0, 1 allowed in the array
    with pytest.raises(ValueError, match="mask values"):
        validate_physical_constraints(state)


# ============================================================
# 5. INITIAL CONDITIONS
# ============================================================

def test_initial_velocity_fields_must_be_finite():
    state = get_valid_state()
    state.fields["U"][0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_physical_constraints(state)


def test_initial_pressure_fields_must_be_finite():
    state = get_valid_state()
    state.fields["P"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_physical_constraints(state)


# ============================================================
# 6. TIME STEP
# ============================================================

def test_time_step_must_be_positive():
    state = get_valid_state()
    state.constants["dt"] = 0.0
    with pytest.raises(ValueError, match="time step"):
        validate_physical_constraints(state)