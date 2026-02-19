# tests/step1/test_parse_boundary_conditions_math.py

import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions

@pytest.fixture
def dummy_grid():
    """Provides a minimal grid dict for the parser."""
    return {"nx": 2, "ny": 2, "nz": 2, "x_max": 1.0, "x_min": 0.0}

def test_parse_valid_bc(dummy_grid):
    bc_list = [{"location": "x_min", "type": "no_slip", "values": {}}]
    parsed = parse_boundary_conditions(bc_list, dummy_grid)
    assert "x_min" in parsed
    assert parsed["x_min"]["type"] == "no_slip"

def test_invalid_location(dummy_grid):
    bc = [{"location": "invalid_loc", "type": "no_slip", "values": {}}]
    with pytest.raises(ValueError, match="(?i)location"):
        parse_boundary_conditions(bc, dummy_grid)

def test_duplicate_location(dummy_grid):
    bc = [
        {"location": "x_min", "type": "no_slip", "values": {}},
        {"location": "x_min", "type": "no_slip", "values": {}}
    ]
    with pytest.raises(ValueError, match="(?i)duplicate"):
        parse_boundary_conditions(bc, dummy_grid)

def test_pressure_requires_p_value(dummy_grid):
    bc = [{"location": "x_max", "type": "pressure", "values": {}}]
    # FIXED: Added (?i) for case-insensitivity to match "Pressure..."
    with pytest.raises(ValueError, match="(?i)pressure"):
        parse_boundary_conditions(bc, dummy_grid)

def test_inflow_requires_components(dummy_grid):
    bc = [{"location": "x_min", "type": "inflow", "values": {"u": 1.0}}] # missing v, w
    with pytest.raises(ValueError, match="(?i)requires velocity component"):
        parse_boundary_conditions(bc, dummy_grid)