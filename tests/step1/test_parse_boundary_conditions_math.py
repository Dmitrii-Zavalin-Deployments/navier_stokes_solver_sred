# tests/step1/test_parse_boundary_conditions_math.py

import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions
from src.step1.types import GridConfig


GRID = GridConfig(
    nx=4, ny=4, nz=4,
    dx=1.0, dy=1.0, dz=1.0,
    x_min=0.0, x_max=4.0,
    y_min=0.0, y_max=4.0,
    z_min=0.0, z_max=4.0,
)


# ---------------------------------------------------------
# Invalid location
# ---------------------------------------------------------

def test_invalid_location():
    bc = [{"location": "diagonal", "type": "no-slip"}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


# ---------------------------------------------------------
# Invalid type
# ---------------------------------------------------------

def test_invalid_type():
    bc = [{"location": "x_min", "type": "warp-drive"}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


# ---------------------------------------------------------
# Duplicate location
# ---------------------------------------------------------

def test_duplicate_location():
    bc = [
        {"location": "x_min", "type": "no-slip"},
        {"location": "x_min", "type": "pressure", "values": {"p": 1.0}},
    ]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


# ---------------------------------------------------------
# Inflow requires velocity components
# ---------------------------------------------------------

def test_inflow_requires_velocity():
    bc = [{"location": "x_min", "type": "inflow", "values": {}}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


# ---------------------------------------------------------
# Pressure BC requires p-value
# ---------------------------------------------------------

def test_pressure_requires_p_value():
    bc = [{"location": "x_max", "type": "pressure", "values": {}}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


# ---------------------------------------------------------
# No-slip must not include pressure
# ---------------------------------------------------------

def test_no_slip_disallows_pressure():
    bc = [{"location": "y_min", "type": "no-slip", "values": {"p": 5.0}}]
    with pytest.raises(ValueError):
        parse_boundary_conditions(bc, GRID)


# ---------------------------------------------------------
# Valid BC normalization
# ---------------------------------------------------------

def test_valid_bc_normalization():
    bc = [{
        "location": "x_min",
        "type": "inflow",
        "values": {"u": 1.0, "v": 2.0, "w": 3.0},
        "comment": "test",
    }]

    table = parse_boundary_conditions(bc, GRID)

    assert "x_min" in table
    entry = table["x_min"]

    assert entry["type"] == "inflow"
    assert entry["u"] == 1.0
    assert entry["v"] == 2.0
    assert entry["w"] == 3.0
    assert entry["comment"] == "test"
