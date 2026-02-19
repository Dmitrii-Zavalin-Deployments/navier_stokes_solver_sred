# tests/step1/test_orchestrate_step1_state.py

import pytest
import copy

from src.step1.orchestrate_step1 import orchestrate_step1_state
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy


# ------------------------------------------------------------
# Test 1 — Input schema validation failure
# ------------------------------------------------------------
def test_step1_input_schema_failure():
    bad = solver_input_schema_dummy()
    bad = copy.deepcopy(bad)

    # Remove a required top-level key
    bad.pop("domain")

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step1_state(bad)

    assert "Input schema validation FAILED" in str(excinfo.value)


# ------------------------------------------------------------
# Test 2 — Physical constraints failure
# ------------------------------------------------------------
def test_step1_physical_constraints_failure():
    bad = solver_input_schema_dummy()
    bad = copy.deepcopy(bad)

    # Invalid density
    bad["fluid_properties"]["density"] = -1.0

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


# ------------------------------------------------------------
# Test 3 — Geometry mask mapping
# ------------------------------------------------------------
def test_step1_geometry_mask_mapping():
    inp = solver_input_schema_dummy()
    state = orchestrate_step1_state(copy.deepcopy(inp))

    # mask must be 2×2×2 (as defined in the dummy)
    assert state.mask.shape == (2, 2, 2)

    # all values must be 1 (fluid)
    assert (state.mask == 1).all()


# ------------------------------------------------------------
# Test 4 — Boundary conditions normalization
# ------------------------------------------------------------
def test_step1_boundary_conditions():
    inp = solver_input_schema_dummy()
    state = orchestrate_step1_state(copy.deepcopy(inp))

    assert hasattr(state, "boundary_conditions")
    assert len(state.boundary_conditions) >= 1


# ------------------------------------------------------------
# Test 5 — Derived constants correctness
# ------------------------------------------------------------
def test_step1_derived_constants():
    inp = solver_input_schema_dummy()
    state = orchestrate_step1_state(copy.deepcopy(inp))

    c = state.constants

    assert c.rho == inp["fluid_properties"]["density"]
    assert c.mu == inp["fluid_properties"]["viscosity"]
    assert c.dt == inp["simulation_parameters"]["time_step"]
    assert c.inv_dx == pytest.approx(1.0 / c.dx)


# ------------------------------------------------------------
# Test 6 — Output schema validation failure
# ------------------------------------------------------------
def test_step1_output_schema_failure(monkeypatch):
    import src.step1.orchestrate_step1 as mod

    real = mod.assemble_simulation_state

    def broken(*args, **kwargs):
        state = real(*args, **kwargs)
        # Remove a required attribute to break schema validation
        state.grid = None
        return state

    monkeypatch.setattr(mod, "assemble_simulation_state", broken)

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step1_state(solver_input_schema_dummy())

    assert "Output schema validation FAILED" in str(excinfo.value)


# ------------------------------------------------------------
# Test 7 — Full happy path
# ------------------------------------------------------------
def test_step1_happy_path():
    inp = solver_input_schema_dummy()
    state = orchestrate_step1_state(copy.deepcopy(inp))

    # Basic structural checks
    assert state.grid["nx"] == inp["domain"]["nx"]
    assert state.fields["P"].shape == (
        inp["domain"]["nx"],
        inp["domain"]["ny"],
        inp["domain"]["nz"],
    )
    assert state.mask.shape == (
        inp["domain"]["nx"],
        inp["domain"]["ny"],
        inp["domain"]["nz"],
    )
    assert state.constants.dt > 0
