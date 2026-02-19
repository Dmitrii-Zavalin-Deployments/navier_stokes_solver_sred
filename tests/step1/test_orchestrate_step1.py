# tests/step1/test_orchestrate_step1_state.py

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
    bad = copy.deepcopy(bad)

    # Remove a required top-level key defined in solver_input_schema.json
    bad.pop("grid")

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step1_state(bad)

    assert "Input schema validation FAILED" in str(excinfo.value)


# ------------------------------------------------------------
# Test 2 — Physical constraints failure
# ------------------------------------------------------------
def test_step1_physical_constraints_failure():
    """Verifies that physically impossible values trigger a ValueError."""
    bad = solver_input_schema_dummy()
    bad = copy.deepcopy(bad)

    # Invalid density (must be > 0)
    bad["fluid_properties"]["density"] = -1.0

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


# ------------------------------------------------------------
# Test 3 — Geometry mask mapping
# ------------------------------------------------------------
def test_step1_geometry_mask_mapping():
    """Checks that the flat mask list is correctly reshaped into a 3D array."""
    inp = solver_input_schema_dummy()
    state = orchestrate_step1_state(copy.deepcopy(inp))

    # mask must be 2×2×2 (as defined in the dummy)
    assert state.mask.shape == (2, 2, 2)
    assert isinstance(state.mask, np.ndarray)

    # In your dummy, you have a specific pattern of -1, 0, 1
    # We check that at least one fluid cell (0) and boundary cell (-1/1) exists
    assert 0 in state.mask
    assert 1 in state.mask or -1 in state.mask


# ------------------------------------------------------------
# Test 4 — Boundary conditions normalization
# ------------------------------------------------------------
def test_step1_boundary_conditions():
    """Ensures the BC table is parsed and attached to the state."""
    inp = solver_input_schema_dummy()
    state = orchestrate_step1_state(copy.deepcopy(inp))

    assert hasattr(state, "boundary_conditions")
    # Even if empty, it should be a dictionary
    assert isinstance(state.boundary_conditions, (dict, list))


# ------------------------------------------------------------
# Test 5 — Derived constants correctness
# ------------------------------------------------------------
def test_step1_derived_constants():
    """Verifies the math for constants like density and inverse grid spacing."""
    inp = solver_input_schema_dummy()
    state = orchestrate_step1_state(copy.deepcopy(inp))

    # Constants is usually a dictionary in the SolverState object
    c = state.constants

    assert c["rho"] == inp["fluid_properties"]["density"]
    assert c["mu"] == inp["fluid_properties"]["viscosity"]
    assert c["dt"] == inp["simulation_parameters"]["time_step"]
    
    # Check derived math: inv_dx = 1.0 / dx
    if "inv_dx" in c:
        assert c["inv_dx"] == pytest.approx(1.0 / c["dx"])


# ------------------------------------------------------------
# Test 6 — Output schema validation failure
# ------------------------------------------------------------
def test_step1_output_schema_failure(monkeypatch):
    """Mocks the state assembler to produce a bad object, triggering validation failure."""
    import src.step1.orchestrate_step1 as mod

    real = mod.assemble_simulation_state

    def broken(*args, **kwargs):
        state = real(*args, **kwargs)
        # Corrupt the state object by removing required grid metadata
        state.grid = {} 
        return state

    monkeypatch.setattr(mod, "assemble_simulation_state", broken)

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step1_state(solver_input_schema_dummy())

    assert "Output schema validation FAILED" in str(excinfo.value)


# ------------------------------------------------------------
# Test 7 — Full happy path
# ------------------------------------------------------------
def test_step1_happy_path():
    """Final end-to-end check for Step 1 pipeline."""
    inp = solver_input_schema_dummy()
    state = orchestrate_step1_state(copy.deepcopy(inp))

    # Basic structural checks
    assert state.grid["nx"] == inp["grid"]["nx"]
    
    # Check field shapes (P=nx,ny,nz while U,V,W are staggered)
    nx, ny, nz = state.grid["nx"], state.grid["ny"], state.grid["nz"]
    assert state.fields["P"].shape == (nx, ny, nz)
    assert state.fields["U"].shape == (nx + 1, ny, nz)
    
    assert state.constants["dt"] > 0