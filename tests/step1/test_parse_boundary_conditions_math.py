# tests/step1/test_parse_boundary_conditions_math.py

import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_grid():
    """Provides the canonical grid metadata from the dummy."""
    return solver_input_schema_dummy()["grid"]

# ---------------------------------------------------------
# Invalid location
# ---------------------------------------------------------

def test_invalid_location(dummy_grid):
    bc = [{"location": "diagonal", "type": "no-slip"}]
    with pytest.raises(ValueError, match="location"):
        parse_boundary_conditions(bc, dummy_grid)


# ---------------------------------------------------------
# Invalid type
# ---------------------------------------------------------

def test_invalid_type(dummy_grid):
    bc = [{"location": "x_min", "type": "warp-drive"}]
    with pytest.raises(ValueError, match="type"):
        parse_boundary_conditions(bc, dummy_grid)


# ---------------------------------------------------------
# Duplicate location
# ---------------------------------------------------------

def test_duplicate_location(dummy_grid):
    bc = [
        {"location": "x_min", "type": "no-slip"},
        {"location": "x_min", "type": "pressure", "values": {"p": 1.0}},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        parse_boundary_conditions(bc, dummy_grid)


# ---------------------------------------------------------
# Inflow requires velocity components
# ---------------------------------------------------------

def test_inflow_requires_velocity(dummy_grid):
    bc = [{"location": "x_min", "type": "inflow", "values": {}}]
    with pytest.raises(ValueError, match="velocity"):
        parse_boundary_conditions(bc, dummy_grid)


# ---------------------------------------------------------
# Pressure BC requires p-value
# ---------------------------------------------------------

def test_pressure_requires_p_value(dummy_grid):
    bc = [{"location": "x_max", "type": "pressure", "values": {}}]
    with pytest.raises(ValueError, match="pressure"):
        parse_boundary_conditions(bc, dummy_grid)


# ---------------------------------------------------------
# No-slip must not include pressure
# ---------------------------------------------------------

def test_no_slip_disallows_pressure(dummy_grid):
    bc = [{"location": "y_min", "type": "no-slip", "values": {"p": 5.0}}]
    with pytest.raises(ValueError, match="pressure"):
        parse_boundary_conditions(bc, dummy_grid)


# ---------------------------------------------------------
# Valid BC normalization & Object Integration
# ---------------------------------------------------------

def test_valid_bc_normalization(dummy_grid):
    bc_list = [{
        "location": "x_min",
        "type": "inflow",
        "values": {"u": 1.0, "v": 2.0, "w": 3.0},
        "comment": "test",
    }]

    # Act
    bc_table = parse_boundary_conditions(bc_list, dummy_grid)
    state = SolverState(boundary_conditions=bc_table, grid=dummy_grid)

    # Assertions using Object Attribute Access (.boundary_conditions)
    assert "x_min" in state.boundary_conditions
    entry = state.boundary_conditions["x_min"]

    assert entry["type"] == "inflow"
    assert entry["u"] == 1.0
    assert entry["v"] == 2.0
    assert entry["w"] == 3.0
    assert entry["comment"] == "test"