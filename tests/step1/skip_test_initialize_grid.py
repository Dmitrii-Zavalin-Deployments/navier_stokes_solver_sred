# tests/step1/test_initialize_grid.py

import pytest
import numpy as np
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
    # Create a copy of the raw input to use for calculating expected values
    raw_grid = dummy_input["grid"].copy()
    
    # Act: initialize the grid dictionary
    # initialize_grid calculates dx, dy, dz and returns a processed dict
    grid_result = initialize_grid(dummy_input["grid"])
    
    # Re-inject extents if the function stripped them, mirroring orchestrator behavior
    grid_result.update({
        "x_min": raw_grid.get("x_min"),
        "x_max": raw_grid.get("x_max"),
        "y_min": raw_grid.get("y_min"),
        "y_max": raw_grid.get("y_max"),
        "z_min": raw_grid.get("z_min"),
        "z_max": raw_grid.get("z_max"),
    })
    
    # Wrap in SolverState to verify object-style integration
    state = SolverState(grid=grid_result)

    # 1. Assertions on Dimensions
    assert state.grid["nx"] == raw_grid["nx"]
    assert state.grid["ny"] == raw_grid["ny"]
    assert state.grid["nz"] == raw_grid["nz"]

    # 2. Assertions on Spacing (dx, dy, dz)
    # Spacing calculation: (max - min) / counts
    expected_dx = (raw_grid["x_max"] - raw_grid["x_min"]) / raw_grid["nx"]
    expected_dy = (raw_grid["y_max"] - raw_grid["y_min"]) / raw_grid["ny"]
    expected_dz = (raw_grid["z_max"] - raw_grid["z_min"]) / raw_grid["nz"]

    assert state.grid["dx"] == pytest.approx(expected_dx)
    assert state.grid["dy"] == pytest.approx(expected_dy)
    assert state.grid["dz"] == pytest.approx(expected_dz)

def test_initialize_grid_invalid_extents(dummy_input):
    """
    Verifies that zero-width or negative-width grids are rejected.
    """
    grid = dummy_input["grid"].copy()
    
    # Override x_max to be equal to x_min (zero extent)
    grid["x_max"] = grid["x_min"]

    with pytest.raises(ValueError, match="x_max"):
        initialize_grid(grid)

def test_initialize_grid_types(dummy_input):
    """Ensures calculated spacing values are floats and grid counts are ints."""
    grid_processed = initialize_grid(dummy_input["grid"])
    
    assert isinstance(grid_processed["nx"], int)
    assert isinstance(grid_processed["dx"], (float, np.float64))