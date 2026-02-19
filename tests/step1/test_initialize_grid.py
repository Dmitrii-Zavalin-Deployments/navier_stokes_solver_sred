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
    when provided with valid domain parameters from the dummy.
    """
    domain = dummy_input["domain"]
    
    # Act: initialize the grid dictionary
    grid = initialize_grid(domain)
    
    # Wrap in SolverState to verify object-style integration
    state = SolverState(grid=grid)

    # 1. Assertions on Dimensions
    assert state.grid["nx"] == domain["nx"]
    assert state.grid["ny"] == domain["ny"]
    assert state.grid["nz"] == domain["nz"]

    # 2. Assertions on Spacing (dx, dy, dz)
    # For dummy: (1.0 - 0.0) / 2 = 0.5
    expected_dx = (domain["x_max"] - domain["x_min"]) / domain["nx"]
    assert state.grid["dx"] == pytest.approx(expected_dx)
    assert state.grid["dy"] == pytest.approx(expected_dx)
    assert state.grid["dz"] == pytest.approx(expected_dx)

def test_initialize_grid_invalid_extents(dummy_input):
    """
    Verifies that zero-width or negative-width domains are rejected.
    """
    domain = dummy_input["domain"]
    
    # Override x_max to be equal to x_min (zero extent)
    domain["x_max"] = domain["x_min"]

    with pytest.raises(ValueError, match="x_max"):
        initialize_grid(domain)

def test_initialize_grid_types(dummy_input):
    """Ensures calculated spacing values are floats and grid counts are ints."""
    grid = initialize_grid(dummy_input["domain"])
    
    assert isinstance(grid["nx"], int)
    assert isinstance(grid["dx"], float)