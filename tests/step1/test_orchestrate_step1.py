# tests/step1/test_orchestrate_step1.py

import pytest
import copy
import numpy as np

from src.step1.orchestrate_step1 import orchestrate_step1_state
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

# ------------------------------------------------------------
# Test 1 — Input schema validation failure
# ------------------------------------------------------------
def test_step1_input_schema_failure():
    """Verifies that missing top-level JSON keys trigger a RuntimeError."""
    bad = solver_input_schema_dummy()
    # Deep copy to avoid mutating the dummy for other tests
    bad = copy.deepcopy(bad)

    # Remove a required top-level key defined in solver_input_schema.json
    bad.pop("grid")

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step1_state(bad)

    assert "Input schema validation FAILED" in str(excinfo.value)


# ------------------------------------------------------------
# Test 2 — Physical constraints failure (Schema Level)
# ------------------------------------------------------------
def test_step1_physical_constraints_failure():
    """
    Verifies that physically impossible values (like negative density) 
    are caught by the JSON validator's exclusiveMinimum rules.
    """
    bad = solver_input_schema_dummy()
    bad = copy.deepcopy(bad)

    # Invalid density (must be > 0 per schema)
    bad["fluid_properties"]["density"] = -1.0

    with pytest.raises(RuntimeError, match="Input schema validation FAILED"):
        orchestrate_step1_state(bad)


# ------------------------------------------------------------
# Test 3 — Geometry mask mapping
# ------------------------------------------------------------
def test_step1_geometry_mask_mapping():
    """Checks that the flat mask list is correctly reshaped into a 3D array."""
    inp = solver_input_schema_dummy()
    state = orchestrate_step1_state(copy.deepcopy(inp))

    # Mask must be 2×2×2 (per dummy_grid dimensions)
    assert state.mask.shape == (2, 2, 2)
    assert isinstance(state.mask, np.ndarray)
    assert np.issubdtype(state.mask.dtype, np.integer)


# ------------------------------------------------------------
# Test 4 — Boundary conditions normalization
# ------------------------------------------------------------
def test_step1_boundary_conditions():
    """Ensures the BC table is parsed and attached to the state."""
    inp = solver_input_schema_dummy()
    state = orchestrate_step1_state(copy.deepcopy(inp))

    assert hasattr(state, "boundary_conditions")
    assert isinstance(state.boundary_conditions, dict)


# ------------------------------------------------------------
# Test 5 — Derived constants correctness
# ------------------------------------------------------------
def test_step1_derived_constants():
    """Verifies derived properties like inverse grid spacing and rho/mu."""
    inp = solver_input_schema_dummy()
    state = orchestrate_step1_state(copy.deepcopy(inp))

    c = state.constants

    assert c["rho"] == inp["fluid_properties"]["density"]
    assert c["mu"] == inp["fluid_properties"]["viscosity"]
    assert c["dt"] == inp["simulation_parameters"]["time_step"]
    
    # Check derived math if implementation uses inverse spacing
    if "inv_dx" in c:
        assert c["inv_dx"] == pytest.approx(1.0 / c["dx"])


# ------------------------------------------------------------
# Test 6 — Internal Structural failure (Monkeypatch)
# ------------------------------------------------------------
def test_step1_output_schema_failure(monkeypatch):
    """
    Forced failure: Verifies the second gate (Physical Validation) 
    triggers if the internal assembler produces a broken state.
    """
    import src.step1.orchestrate_step1 as mod

    real = mod.assemble_simulation_state

    def broken(*args, **kwargs):
        state = real(*args, **kwargs)
        # Corrupt the object state after assembly but before final validation
        state.grid = {} 
        return state

    monkeypatch.setattr(mod, "assemble_simulation_state", broken)

    with pytest.raises(ValueError, match="Incomplete Grid Definition"):
        orchestrate_step1_state(solver_input_schema_dummy())


# ------------------------------------------------------------
# Test 7 — Full happy path (Staggered Grid Verification)
# ------------------------------------------------------------
def test_step1_happy_path():
    """Final end-to-end check for Step 1 pipeline including staggered shapes."""
    inp = solver_input_schema_dummy()
    state = orchestrate_step1_state(copy.deepcopy(inp))

    # Grid parameters
    nx, ny, nz = state.grid["nx"], state.grid["ny"], state.grid["nz"]
    
    # 1. Check Field Allocation (Staggered Grid logic)
    # Pressure lives at cell centers: (nx, ny, nz)
    assert state.fields["P"].shape == (nx, ny, nz)
    # U-velocity lives on X-faces: (nx+1, ny, nz)
    assert state.fields["U"].shape == (nx + 1, ny, nz)
    # V-velocity lives on Y-faces: (nx, ny+1, nz)
    assert state.fields["V"].shape == (nx, ny + 1, nz)
    # W-velocity lives on Z-faces: (nx, ny, nz+1)
    assert state.fields["W"].shape == (nx, ny, nz + 1)
    
    assert state.constants["dt"] > 0