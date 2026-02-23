# tests/step1/test_initialize_grid.py

import pytest
import numpy as np
from src.step1.initialize_grid import initialize_grid
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_input():
    """Provides the canonical JSON-safe dummy input (Section 5 Compliance)."""
    return solver_input_schema_dummy()

def test_initialize_grid_basic(dummy_input):
    """
    Verifies functional contract: converts raw grid params into processed state.
    """
    raw_grid = dummy_input["grid"].copy()
    
    # Act: initialize the grid dictionary
    grid_result = initialize_grid(dummy_input["grid"])
    
    # Re-inject extents for metadata completeness (simulating orchestrator logic)
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

    # 1. Assertions on Dimensions (Integrity)
    assert state.grid["nx"] == raw_grid["nx"]
    assert state.grid["ny"] == raw_grid["ny"]
    assert state.grid["nz"] == raw_grid["nz"]

    # 2. Assertions on Spacing (Precision)
    expected_dx = (raw_grid["x_max"] - raw_grid["x_min"]) / raw_grid["nx"]
    expected_dy = (raw_grid["y_max"] - raw_grid["y_min"]) / raw_grid["ny"]
    expected_dz = (raw_grid["z_max"] - raw_grid["z_min"]) / raw_grid["nz"]

    assert state.grid["dx"] == pytest.approx(expected_dx)
    assert state.grid["dy"] == pytest.approx(expected_dy)
    assert state.grid["dz"] == pytest.approx(expected_dz)

def test_initialize_grid_invalid_extents(dummy_input):
    """Verifies rejection of unphysical zero-width domains."""
    grid = dummy_input["grid"].copy()
    grid["x_max"] = grid["x_min"]

    with pytest.raises(ValueError, match="(?i)x_max"):
        initialize_grid(grid)

def test_initialize_grid_types(dummy_input):
    """Ensures strict typing for downstream stability."""
    grid_processed = initialize_grid(dummy_input["grid"])
    
    # Grid counts must be integers for array indexing
    assert isinstance(grid_processed["nx"], int)
    # Spacing must be floats for continuous physics
    assert isinstance(grid_processed["dx"], (float, np.float64, np.float32))