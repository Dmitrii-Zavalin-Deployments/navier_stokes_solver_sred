# tests/step1/test_initialize_grid_full.py

import pytest
from src.step1.initialize_grid import initialize_grid
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def base_input():
    """Provides the canonical dummy input."""
    return solver_input_schema_dummy()

def test_invalid_y_extent(base_input):
    """Verifies that a zero-thickness Y-domain triggers a ValueError."""
    domain = base_input["domain"]
    domain.update({
        "y_min": 1.0, 
        "y_max": 1.0  # Invalid: y_max must be > y_min
    })

    with pytest.raises(ValueError, match="y_max"):
        initialize_grid(domain)

def test_invalid_z_extent(base_input):
    """Verifies that a zero-thickness Z-domain triggers a ValueError."""
    domain = base_input["domain"]
    domain.update({
        "z_min": 2.0, 
        "z_max": 2.0  # Invalid: z_max must be > z_min
    })

    with pytest.raises(ValueError, match="z_max"):
        initialize_grid(domain)

def test_grid_initialization_in_state(base_input):
    """
    Integration test: Verifies that initialize_grid output is correctly 
    stored in the SolverState.grid attribute.
    """
    domain_params = base_input["domain"]
    
    # Act
    grid_dict = initialize_grid(domain_params)
    state = SolverState(grid=grid_dict)

    # 1. Check Object-style access
    assert hasattr(state, "grid")
    assert isinstance(state.grid, dict)

    # 2. Check calculated values in the grid dict
    # dx = (x_max - x_min) / nx = (1.0 - 0.0) / 2 = 0.5
    assert state.grid["dx"] == 0.5
    assert state.grid["nx"] == domain_params["nx"]