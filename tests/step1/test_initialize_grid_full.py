# tests/step1/test_initialize_grid_full.py

import pytest
from src.step1.initialize_grid import initialize_grid
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def base_input():
    """Provides the canonical dummy input (Section 5 Compliance)."""
    return solver_input_schema_dummy()

def test_invalid_x_extent(base_input):
    """Verifies that a zero-thickness X-grid triggers a ValueError."""
    grid = base_input["grid"]
    grid.update({"x_min": 1.0, "x_max": 1.0})
    with pytest.raises(ValueError, match="x_max"):
        initialize_grid(grid)

def test_invalid_y_extent(base_input):
    """Verifies that a zero-thickness Y-grid triggers a ValueError."""
    grid = base_input["grid"]
    grid.update({"y_min": 1.0, "y_max": 1.0})
    with pytest.raises(ValueError, match="y_max"):
        initialize_grid(grid)

def test_invalid_z_extent(base_input):
    """Verifies that a zero-thickness Z-grid triggers a ValueError."""
    grid = base_input["grid"]
    grid.update({"z_min": 2.0, "z_max": 2.0})
    with pytest.raises(ValueError, match="z_max"):
        initialize_grid(grid)

def test_grid_initialization_in_state(base_input):
    """
    Integration test: Verifies that initialize_grid output is correctly 
    stored in the SolverState.grid attribute.
    """
    grid_params = base_input["grid"]
    
    # Act
    grid_dict = initialize_grid(grid_params)
    state = SolverState(grid=grid_dict)

    # 1. Check Object-style access
    assert hasattr(state, "grid")
    assert isinstance(state.grid, dict)

    # 2. Check calculated values (e.g., dx = (1.0 - 0.0) / 2 = 0.5)
    assert state.grid["dx"] == 0.5
    assert state.grid["nx"] == grid_params["nx"]
    assert "dy" in state.grid
    assert "dz" in state.grid