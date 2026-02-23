# tests/step1/test_initialize_grid_math.py

import pytest
from src.step1.initialize_grid import initialize_grid
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def base_input():
    """Provides the canonical dummy input (Section 5 Compliance)."""
    return solver_input_schema_dummy()

def test_correct_grid_spacing_computation(base_input):
    """Verifies mathematical correctness of dx, dy, dz calculations."""
    grid = base_input["grid"]
    grid.update({
        "nx": 4, "ny": 2, "nz": 1,
        "x_min": 0.0, "x_max": 8.0,   
        "y_min": -1.0, "y_max": 3.0,  
        "z_min": 2.0, "z_max": 6.0,   
    })

    grid_data = initialize_grid(grid)
    state = SolverState(grid=grid_data)

    # Spacing is Delta L / N (Standard uniform discretization logic)
    assert state.grid["dx"] == pytest.approx(grid["x_max"] / grid["nx"])
    assert state.grid["dy"] == pytest.approx((grid["y_max"] - grid["y_min"]) / grid["ny"])



def test_missing_required_keys_raise_keyerror(base_input):
    """Ensures a ValueError is raised if any mandatory grid parameter is missing."""
    base_grid = base_input["grid"]

    for key in list(base_grid.keys()):
        bad_grid = dict(base_grid)
        del bad_grid[key]
        # Constitutional Role: Validates that Phase F Data Intake wraps KeyErrors.
        with pytest.raises(ValueError, match="Grid initialization failed"):
            initialize_grid(bad_grid)

def test_grid_dimensions_must_be_positive(base_input):
    """Checks that non-positive nx, ny, nz raise ValueErrors."""
    grid = base_input["grid"]
    
    # Check zero and negative for nx
    for val in [0, -5]:
        grid["nx"] = val
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            initialize_grid(grid)

def test_extents_must_be_finite(base_input):
    """Safety: Ensures grid boundaries are not Inf or NaN to prevent solver crash."""
    grid = base_input["grid"]
    for bad in [float("inf"), float("nan")]:
        grid["x_min"] = bad
        with pytest.raises(ValueError, match="(?i)finite"):
            initialize_grid(grid)

def test_extents_must_be_ordered_correctly(base_input):
    """Logic Check: max must be strictly greater than min."""
    grid = base_input["grid"]
    
    # Case: Equal (zero width)
    # FIXED: Matches the actual message: 'Inverted domain: x_max (10.0) must be > x_min (10.0).'
    grid["x_min"], grid["x_max"] = 10.0, 10.0
    with pytest.raises(ValueError, match="Inverted domain"):
        initialize_grid(grid)

    # Case: Reversed (negative width)
    grid["x_max"] = 5.0
    with pytest.raises(ValueError, match="Inverted domain"):
        initialize_grid(grid)

def test_dx_dy_dz_calculated_correctly_in_state(base_input):
    """Integration: Ensures SolverState correctly stores high-precision float spacing."""
    grid_data = initialize_grid(base_input["grid"])
    state = SolverState(grid=grid_data)
    
    assert state.grid["dx"] > 0
    assert isinstance(state.grid["dx"], float)