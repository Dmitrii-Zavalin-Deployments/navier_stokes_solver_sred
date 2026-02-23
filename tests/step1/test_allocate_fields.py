import numpy as np
import pytest

# Core imports
from src.step1.allocate_fields import allocate_fields
from src.solver_state import SolverState

# Dummy helper import
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_input():
    """Provides the canonical JSON-safe dummy input."""
    return solver_input_schema_dummy()

def test_cell_centered_and_staggered_field_shapes(dummy_input):
    """
    Verifies field allocation shapes for Arakawa C-grid staggering.
    """
    # Standardize the grid dict for the allocator
    grid_params = dummy_input["grid"]
    grid = {
        "nx": grid_params["nx"],
        "ny": grid_params["ny"],
        "nz": grid_params["nz"]
    }

    # 1. Test the allocation function directly (raw dictionary output)
    fields = allocate_fields(grid)
    
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    
    # Assert Staggered Parity:
    # P is cell-centered (nx, ny, nz)
    # U is on x-faces (nx+1, ny, nz)
    # V is on y-faces (nx, ny+1, nz)
    # W is on z-faces (nx, ny, nz+1)
    assert fields["P"].shape == (nx, ny, nz)
    assert fields["U"].shape == (nx + 1, ny, nz)
    assert fields["V"].shape == (nx, ny + 1, nz)
    assert fields["W"].shape == (nx, ny, nz + 1)

    # 2. Test the SolverState attribute mapping
    # We pass the dictionary; SolverState should map these to internal attributes
    state = SolverState(fields=fields, grid=grid)

    # Verify direct attribute access (The Production standard)
    assert state.pressure.shape == (nx, ny, nz)
    assert state.velocity_u.shape == (nx + 1, ny, nz)
    assert state.velocity_v.shape == (nx, ny + 1, nz)
    assert state.velocity_w.shape == (nx, ny, nz + 1)

def test_field_initialization_to_zero(dummy_input):
    """Ensures all allocated fields start at zero (Pre-IC state)."""
    grid_params = dummy_input["grid"]
    grid = {"nx": grid_params["nx"], "ny": grid_params["ny"], "nz": grid_params["nz"]}
    fields = allocate_fields(grid)
    state = SolverState(fields=fields)

    # Check via mapped attributes to ensure the pointers are correct
    assert np.all(state.pressure == 0.0)
    assert np.all(state.velocity_u == 0.0)
    assert np.all(state.velocity_v == 0.0)
    assert np.all(state.velocity_w == 0.0)