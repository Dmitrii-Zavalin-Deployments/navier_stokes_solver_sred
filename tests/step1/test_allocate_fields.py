# tests/step1/test_allocate_fields.py

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
    Verifies field allocation shapes based on the canonical dummy input.
    """
    domain = dummy_input["domain"]
    grid = {
        "nx": domain["nx"],
        "ny": domain["ny"],
        "nz": domain["nz"]
    }

    # 1. Test the allocation function directly (returns dict)
    fields = allocate_fields(grid)
    
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    assert fields["P"].shape == (nx, ny, nz)
    assert fields["U"].shape == (nx + 1, ny, nz)
    assert fields["V"].shape == (nx, ny + 1, nz)
    assert fields["W"].shape == (nx, ny, nz + 1)

    # 2. Test how it behaves inside the SolverState object
    state = SolverState(fields=fields, grid=grid)

    # NEW: Using Object-style attribute access
    assert isinstance(state.fields, dict)
    assert state.fields["P"].shape == (nx, ny, nz)
    
    # Verify we can access staggered components correctly through the object
    assert state.fields["U"].shape[0] == nx + 1
    assert state.fields["V"].shape[1] == ny + 1
    assert state.fields["W"].shape[2] == nz + 1

def test_field_initialization_to_zero(dummy_input):
    """Ensures all allocated fields start at zero to avoid garbage values."""
    domain = dummy_input["domain"]
    grid = {"nx": domain["nx"], "ny": domain["ny"], "nz": domain["nz"]}
    fields = allocate_fields(grid)
    state = SolverState(fields=fields)

    for field_name in ["P", "U", "V", "W"]:
        arr = state.fields[field_name]
        assert np.all(arr == 0.0), f"Field {field_name} must be initialized to 0.0"