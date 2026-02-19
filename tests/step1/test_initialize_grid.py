# tests/step1/test_initialize_grid.py

import pytest
from src.step1.initialize_grid import initialize_grid
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_input():
    """Provides the canonical JSON-safe dummy input."""
    return solver_input_schema_dummy()

def test_initialize_grid_basic(dummy_input):
    """
    Verifies that the grid is initialized with correct dimensions and spacing
    when provided with valid grid parameters from the dummy.
    """
    grid = dummy_input["grid"]
    
    # Act: initialize the grid dictionary
    grid = initialize_grid(grid)
    
    # Wrap in SolverState to verify object-style integration
    state = SolverState(grid=grid)

    # 1. Assertions on Dimensions
    assert state.grid["nx"] == grid["nx"]
    assert state.grid["ny"] == grid["ny"]
    assert state.grid["nz"] == grid["nz"]

    # 2. Assertions on Spacing (dx, dy, dz)
    # For dummy: (1.0 - 0.0) / 2 = 0.5
    expected_dx = (grid["x_max"] - grid["x_min"]) / grid["nx"]
    assert state.grid["dx"] == pytest.approx(expected_dx)
    assert state.grid["dy"] == pytest.approx(expected_dx)
    assert state.grid["dz"] == pytest.approx(expected_dx)

def test_initialize_grid_invalid_extents(dummy_input):
    """
    Verifies that zero-width or negative-width domains are rejected.
    """
    grid = dummy_input["grid"]
    
    # Override x_max to be equal to x_min (zero extent)
    grid["x_max"] = grid["x_min"]

    with pytest.raises(ValueError, match="x_max"):
        initialize_grid(grid)

def test_initialize_grid_types(dummy_input):
    """Ensures calculated spacing values are floats and grid counts are ints."""
    grid = initialize_grid(dummy_input["grid"])
    
    assert isinstance(grid["nx"], int)
    assert isinstance(grid["dx"], float)