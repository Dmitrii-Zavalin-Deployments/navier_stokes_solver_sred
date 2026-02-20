# tests/step1/test_boundary_logic_compliance.py

import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_grid():
    """Provides the canonical grid metadata from the dummy."""
    return solver_input_schema_dummy()["grid"]

# ---------------------------------------------------------
# Theory Compliance Tests (Error Handling)
# ---------------------------------------------------------

def test_invalid_location_rejected(dummy_grid):
    """Compliance: Validates that location strings must match the schema exactly."""
    bc = [{"location": "diagonal", "type": "no-slip"}]
    with pytest.raises(ValueError, match="(?i)location"):
        parse_boundary_conditions(bc, dummy_grid)

def test_invalid_type_rejected(dummy_grid):
    """Compliance: Rejects BC types not defined in the Enum list."""
    bc = [{"location": "x_min", "type": "warp-drive"}]
    with pytest.raises(ValueError, match="(?i)type"):
        parse_boundary_conditions(bc, dummy_grid)

def test_duplicate_location_rejected(dummy_grid):
    """Compliance: Prevents mathematical ambiguity by forbidding duplicate face definitions."""
    bc = [
        {"location": "x_min", "type": "no-slip"},
        {"location": "x_min", "type": "pressure", "values": {"p": 1.0}},
    ]
    with pytest.raises(ValueError, match="(?i)duplicate"):
        parse_boundary_conditions(bc, dummy_grid)

def test_inflow_action_requires_numerical_uvw(dummy_grid):
    """Action Item: Validates that an inflow BC provides numerical u, v, and w."""
    # Case: Missing components
    bc_missing = [{"location": "x_min", "type": "inflow", "values": {"u": 1.0}}]
    with pytest.raises(ValueError, match="(?i)u, v, and w"):
        parse_boundary_conditions(bc_missing, dummy_grid)
        
    # Case: Non-numerical value
    bc_bad_type = [{"location": "x_min", "type": "inflow", "values": {"u": 1, "v": 0, "w": "fast"}}]
    with pytest.raises(ValueError, match="(?i)numerical"):
        parse_boundary_conditions(bc_bad_type, dummy_grid)

def test_pressure_action_requires_numerical_p(dummy_grid):
    """Action Item: Validates that a pressure BC provides a numerical p."""
    bc = [{"location": "x_max", "type": "pressure", "values": {"u": 0.0}}]
    with pytest.raises(ValueError, match="(?i)numerical p"):
        parse_boundary_conditions(bc, dummy_grid)

def test_static_bc_disallows_wrong_values(dummy_grid):
    """Compliance: Ensure no-slip doesn't accidentally carry pressure values (Debt prevention)."""
    bc = [{"location": "y_min", "type": "no-slip", "values": {"p": 5.0}}]
    with pytest.raises(ValueError, match="(?i)pressure"):
        parse_boundary_conditions(bc, dummy_grid)

# ---------------------------------------------------------
# Integration & Normalization Tests
# ---------------------------------------------------------

def test_valid_bc_storage_and_normalization(dummy_grid):
    """
    Verifies that parsed BCs are accessible and values are properly cast to floats.
    Ensures that default values are applied where 'values' is omitted.
    """
    bc_list = [
        {"location": "z_max", "type": "no-slip"},
        {"location": "x_min", "type": "inflow", "values": {"u": 5, "v": 0, "w": 0}}
    ]
    
    # Act
    bc_table = parse_boundary_conditions(bc_list, dummy_grid)
    state = SolverState(boundary_conditions=bc_table, grid=dummy_grid)
    
    # Assert: Basic Structure
    assert "z_max" in state.boundary_conditions
    assert state.boundary_conditions["z_max"]["type"] == "no-slip"
    
    # Assert: Numerical Normalization (int -> float)
    inflow_u = state.boundary_conditions["x_min"]["u"]
    assert isinstance(inflow_u, float)
    assert inflow_u == 5.0
    
    # Assert: Defaults for omitted values (u=0.0 for no-slip)
    assert state.boundary_conditions["z_max"]["u"] == 0.0