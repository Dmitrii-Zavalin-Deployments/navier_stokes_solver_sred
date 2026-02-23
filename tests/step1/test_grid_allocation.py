# tests/step1/test_grid_allocation.py

import pytest
import numpy as np
from src.step1.initialize_grid import initialize_grid
from src.step1.allocate_fields import allocate_fields
from src.step1.assemble_simulation_state import assemble_simulation_state
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_input():
    """Provides the canonical JSON-safe dummy input (Section 5 Compliance)."""
    return solver_input_schema_dummy()

# --- INITIALIZE GRID TESTS ---

def test_initialize_grid_logic_and_math(dummy_input):
    """Verifies functional contract, converts raw params, and checks spacing precision."""
    raw_grid = dummy_input["grid"].copy()
    grid_result = initialize_grid(raw_grid)
    
    # 1. Assertions on Dimensions
    assert grid_result["nx"] == raw_grid["nx"]
    assert isinstance(grid_result["nx"], int)

    # 2. Assertions on Spacing (Precision)
    expected_dx = (raw_grid["x_max"] - raw_grid["x_min"]) / raw_grid["nx"]
    assert grid_result["dx"] == pytest.approx(expected_dx)
    assert isinstance(grid_result["dx"], (float, np.float64))

def test_initialize_grid_guardrails(dummy_input):
    """Triggers guardrails for missing keys, inverted domains, and non-finite values."""
    grid = dummy_input["grid"].copy()

    # Case: Missing keys
    bad_grid = {"nx": 4} 
    with pytest.raises(ValueError, match="Grid initialization failed"):
        initialize_grid(bad_grid)

    # Case: Inverted domain (max <= min)
    grid["x_min"], grid["x_max"] = 10.0, 10.0
    with pytest.raises(ValueError, match="Inverted domain"):
        initialize_grid(grid)

    # Case: Non-positive dimensions
    grid["nx"] = 0
    with pytest.raises(ValueError, match="Grid dimensions must be positive"):
        initialize_grid(grid)

    # Case: Non-finite values
    grid["x_min"] = float("inf")
    with pytest.raises(ValueError, match="(?i)finite"):
        initialize_grid(grid)

# --- ALLOCATE FIELDS TESTS ---

def test_field_shapes_and_staggering(dummy_input):
    """Verifies field allocation shapes for Arakawa C-grid staggering."""
    grid = {"nx": 4, "ny": 4, "nz": 4}
    fields = allocate_fields(grid)
    
    # P is cell-centered (nx, ny, nz)
    # U, V, W are face-centered (staggered +1)
    assert fields["P"].shape == (4, 4, 4)
    assert fields["U"].shape == (5, 4, 4)
    assert fields["V"].shape == (4, 5, 4)
    assert fields["W"].shape == (4, 4, 5)
    
    # Verify SolverState attribute pointers
    state = SolverState(fields=fields, grid=grid)
    assert state.velocity_u.shape == (5, 4, 4)
    assert np.all(state.pressure == 0.0)

def test_allocate_fields_debt_trigger():
    """Target: allocate_fields.py line 24 (Invalid dims)."""
    with pytest.raises(ValueError, match="Invalid grid dimensions"):
        allocate_fields({"nx": 2, "ny": -1, "nz": 2})

# --- ASSEMBLE STATE DEBT TESTS ---

def test_assemble_state_genesis_and_spatial_errors():
    """Triggers Line 53 (Missing Fields) and Line 58 (Mismatched Mask)."""
    grid = {"nx": 4, "ny": 4, "nz": 4}
    constants = {"rho": 1.0, "mu": 0.1}
    mask_valid = np.zeros((4, 4, 4))
    mask_invalid = np.zeros((2, 2, 2))
    
    fields = {
        "U": np.zeros((5, 4, 4)), "V": np.zeros((4, 5, 4)), 
        "W": np.zeros((4, 4, 5)), "P": np.zeros((4, 4, 4))
    }

    # 1. Missing required field 'W'
    incomplete = {k: v for k, v in fields.items() if k != "W"}
    with pytest.raises(KeyError, match="Genesis Error: Required field 'W' missing"):
        assemble_simulation_state({}, grid, incomplete, mask_valid, constants, {}, mask_valid, mask_valid)

    # 2. Spatial Incoherence (Mask shape mismatch)
    with pytest.raises(ValueError, match="Spatial Incoherence"):
        assemble_simulation_state({}, grid, fields, mask_invalid, constants, {}, mask_invalid, mask_invalid)