import pytest
from src.step1.parse_boundary_conditions import parse_boundary_conditions
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_grid():
    return solver_input_schema_dummy()["grid"]

def test_inflow_action_requires_numerical_uvw(dummy_grid):
    """Checks that inflow requires numerical components."""
    bc_missing = [{"location": "x_min", "type": "inflow", "values": {"u": 1.0}}]
    # Updated match to handle specific component error
    with pytest.raises(ValueError, match="requires numeric velocity"):
        parse_boundary_conditions(bc_missing, dummy_grid)

def test_pressure_action_requires_numerical_p(dummy_grid):
    """Checks that pressure requires numerical p."""
    bc = [{"location": "x_max", "type": "pressure", "values": {"u": 0.0}}]
    with pytest.raises(ValueError, match="requires numeric 'p'"):
        parse_boundary_conditions(bc, dummy_grid)

def test_valid_bc_storage_and_normalization(dummy_grid):
    """Verifies 6-face closure and float conversion."""
    bc_list = [
        {"location": "x_min", "type": "inflow", "values": {"u": 5, "v": 0, "w": 0}},
        {"location": "x_max", "type": "outflow"},
        {"location": "y_min", "type": "no-slip"},
        {"location": "y_max", "type": "no-slip"},
        {"location": "z_min", "type": "no-slip"},
        {"location": "z_max", "type": "no-slip"}
    ]
    
    bc_table = parse_boundary_conditions(bc_list, dummy_grid)
    assert len(bc_table) == 6
    assert isinstance(bc_table["x_min"]["u"], float)
    assert bc_table["x_min"]["u"] == 5.0