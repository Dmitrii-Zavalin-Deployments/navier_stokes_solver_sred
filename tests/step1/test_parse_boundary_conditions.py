# tests/step1/test_parse_boundary_conditions.py

import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_grid():
    """Provides the canonical grid metadata from the dummy."""
    return solver_input_schema_dummy()["grid"]

# ---------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------

def test_invalid_location_rejected(dummy_grid):
    bc = [{"location": "diagonal", "type": "no-slip"}]
    # Using (?i) to make the match case-insensitive
    with pytest.raises(ValueError, match="(?i)location"):
        parse_boundary_conditions(bc, dummy_grid)


def test_invalid_type_rejected(dummy_grid):
    bc = [{"location": "x_min", "type": "warp-drive"}]
    with pytest.raises(ValueError, match="(?i)type"):
        parse_boundary_conditions(bc, dummy_grid)


def test_duplicate_location_rejected(dummy_grid):
    bc = [
        {"location": "x_min", "type": "no-slip"},
        {"location": "x_min", "type": "pressure", "values": {"p": 1.0}},
    ]
    with pytest.raises(ValueError, match="(?i)duplicate"):
        parse_boundary_conditions(bc, dummy_grid)


def test_inflow_requires_velocity(dummy_grid):
    bc = [{"location": "x_min", "type": "inflow", "values": {}}]
    with pytest.raises(ValueError, match="(?i)velocity"):
        parse_boundary_conditions(bc, dummy_grid)


def test_pressure_bc_requires_p(dummy_grid):
    bc = [{"location": "x_max", "type": "pressure", "values": {}}]
    # Matches "Pressure" or "pressure"
    with pytest.raises(ValueError, match="(?i)pressure"):
        parse_boundary_conditions(bc, dummy_grid)


def test_no_slip_disallows_pressure(dummy_grid):
    bc = [{"location": "y_min", "type": "no-slip", "values": {"p": 5.0}}]
    with pytest.raises(ValueError, match="(?i)pressure"):
        parse_boundary_conditions(bc, dummy_grid)

# ---------------------------------------------------------
# Success Case & SolverState Integration
# ---------------------------------------------------------

def test_valid_bc_storage_in_state(dummy_grid):
    """Verifies that parsed BCs are accessible via the .boundary_conditions attribute."""
    bc_list = [{
        "location": "z_max",
        "type": "no-slip"
    }]
    
    # Act
    bc_table = parse_boundary_conditions(bc_list, dummy_grid)
    state = SolverState(boundary_conditions=bc_table, grid=dummy_grid)
    
    # Assert
    assert hasattr(state, "boundary_conditions")
    assert state.boundary_conditions["z_max"]["type"] == "no-slip"
    # Ensure values were normalized to floats even if not provided
    assert isinstance(state.boundary_conditions["z_max"]["u"], float)
    assert state.boundary_conditions["z_max"]["u"] == 0.0