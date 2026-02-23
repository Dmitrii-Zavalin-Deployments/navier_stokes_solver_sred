# tests/step1/test_parse_boundary_conditions_math.py

import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_input():
    """Provides the full canonical dummy dictionary."""
    return solver_input_schema_dummy()

def test_parse_canonical_dummy_bc(dummy_input):
    """
    Verifies that the boundary conditions defined in the 
    official solver_input_schema_dummy are valid and parseable.
    """
    grid = dummy_input["grid"]
    bc_list = dummy_input["boundary_conditions"]
    
    parsed = parse_boundary_conditions(bc_list, grid)
    
    # The dummy defines 'x_min' as no-slip
    assert "x_min" in parsed
    assert parsed["x_min"]["type"] == "no-slip"
    assert isinstance(parsed["x_min"]["values"]["u"], float)

def test_invalid_location_override(dummy_input):
    """Verifies error when the dummy location is tampered with."""
    grid = dummy_input["grid"]
    # Intentional corruption of a valid dummy list
    bc_list = [{"location": "center_of_universe", "type": "no-slip", "values": {}}]
    
    with pytest.raises(ValueError, match="(?i)Invalid location"):
        parse_boundary_conditions(bc_list, grid)

def test_duplicate_location_logic(dummy_input):
    """Ensures a location cannot have two BCs, using dummy grid context."""
    grid = dummy_input["grid"]
    bc_list = [
        {"location": "x_min", "type": "no-slip", "values": {}},
        {"location": "x_min", "type": "outflow", "values": {}}
    ]
    with pytest.raises(ValueError, match="(?i)Duplicate boundary condition"):
        parse_boundary_conditions(bc_list, grid)

def test_pressure_validation_against_dummy(dummy_input):
    """Verifies that the dummy's 'x_max' (outflow/pressure) logic is sound."""
    grid = dummy_input["grid"]
    # Overriding dummy list to create an invalid pressure BC
    bc_list = [{"location": "x_max", "type": "pressure", "values": {}}] # Missing 'p'
    
    with pytest.raises(ValueError, match="(?i)requires 'p'"):
        parse_boundary_conditions(bc_list, grid)

def test_inflow_component_completeness(dummy_input):
    """Tests that inflow requires a full 3D velocity vector."""
    grid = dummy_input["grid"]
    bc_list = [{"location": "x_min", "type": "inflow", "values": {"u": 1.0}}]
    
    with pytest.raises(ValueError, match="(?i)requires velocity components"):
        parse_boundary_conditions(bc_list, grid)