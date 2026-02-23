# tests/step1/test_validate_physical_constraints_math.py

import pytest
import numpy as np
from src.step1.validate_physical_constraints import validate_physical_constraints
from src.step1.orchestrate_step1 import orchestrate_step1_state
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def valid_state():
    """
    Provides a fresh, valid SolverState for every test to prevent 
    cross-test pollution of mutated attributes.
    """
    json_input = solver_input_schema_dummy()
    return orchestrate_step1_state(json_input)

# ============================================================
# 1. FLUID PROPERTIES
# ============================================================

def test_density_must_be_positive(valid_state):
    valid_state.constants["rho"] = 0.0  
    with pytest.raises(ValueError, match="(?i)density"):
        validate_physical_constraints(valid_state)

def test_viscosity_must_be_non_negative(valid_state):
    valid_state.constants["mu"] = -1.0
    with pytest.raises(ValueError, match="(?i)viscosity"):
        validate_physical_constraints(valid_state)

# ============================================================
# 2. GRID METRICS
# ============================================================

def test_grid_extents_ordering(valid_state):
    # Logic: x_max <= x_min is a physical impossibility for a volume
    valid_state.grid["x_max"] = valid_state.grid["x_min"]
    with pytest.raises(ValueError, match="must be >"):
        validate_physical_constraints(valid_state)

def test_grid_finiteness(valid_state):
    valid_state.grid["y_max"] = float("inf")
    with pytest.raises(ValueError, match="(?i)finite"):
        validate_physical_constraints(valid_state)

def test_grid_counts_positivity(valid_state):
    valid_state.grid["nz"] = -1
    with pytest.raises(ValueError, match="nz"):
        validate_physical_constraints(valid_state)

# ============================================================
# 3. MASK & FIELD INTEGRITY
# ============================================================

def test_mask_shape_mismatch(valid_state):
    # If the grid is 2x2x2, the mask must be (2,2,2)
    valid_state.mask = np.zeros((10, 10, 10))
    with pytest.raises(ValueError, match="(?i)shape"):
        validate_physical_constraints(valid_state)

def test_mask_value_compliance(valid_state):
    # Only {-1, 0, 1} are allowed
    valid_state.mask[0, 0, 0] = 42 
    with pytest.raises(ValueError, match="invalid entries"):
        validate_physical_constraints(valid_state)

def test_field_numerical_stability(valid_state):
    # Checks for NaNs in velocity fields during IC intake
    valid_state.fields["V"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="(?i)finite"):
        validate_physical_constraints(valid_state)

# ============================================================
# 4. TEMPORAL CONSTRAINTS
# ============================================================

def test_time_step_positivity(valid_state):
    valid_state.constants["dt"] = -0.001
    with pytest.raises(ValueError, match="(?i)time step"):
        validate_physical_constraints(valid_state)