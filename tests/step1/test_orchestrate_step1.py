# tests/step1/test_orchestrate_step1_state.py

import pytest
import copy

from src.step1.orchestrate_step1 import orchestrate_step1_state
from tests.helpers.minimal_step1_input import MINIMAL_VALID_INPUT


# ------------------------------------------------------------
# Test 1 — Input schema validation failure
# ------------------------------------------------------------
def test_step1_input_schema_failure():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad.pop("domain_definition")  # required key

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step1_state(bad)

    assert "Input schema validation FAILED" in str(excinfo.value)


# ------------------------------------------------------------
# Test 2 — Physical constraints failure
# ------------------------------------------------------------
def test_step1_physical_constraints_failure():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["fluid_properties"]["density"] = -1.0  # invalid

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


# ------------------------------------------------------------
# Test 3 — Geometry mask mapping
# ------------------------------------------------------------
def test_step1_geometry_mask_mapping():
    state = orchestrate_step1_state(copy.deepcopy(MINIMAL_VALID_INPUT))

    # mask must be 2×2×2
    assert state.mask.shape == (2, 2, 2)

    # all values must be 1 (fluid)
    assert (state.mask == 1).all()


# ------------------------------------------------------------
# Test 4 — Boundary conditions normalization
# ------------------------------------------------------------
def test_step1_boundary_conditions():
    state = orchestrate_step1_state(copy.deepcopy(MINIMAL_VALID_INPUT))

    # boundary_conditions must exist and be non-empty
    assert hasattr(state, "boundary_conditions")
    assert len(state.boundary_conditions) >= 1


# ------------------------------------------------------------
# Test 5 — Derived constants correctness
# ------------------------------------------------------------
def test_step1_derived_constants():
    state = orchestrate_step1_state(copy.deepcopy(MINIMAL_VALID_INPUT))

    c = state.constants

    assert c.rho == 1.0
    assert c.mu == 0.1
    assert c.dt > 0
    assert c.inv_dx == pytest.approx(1.0 / c.dx)


# ------------------------------------------------------------
# Test 6 — Output schema validation failure
# ------------------------------------------------------------
def test_step1_output_schema_failure(monkeypatch):
    import src.step1.orchestrate_step1 as mod

    real = mod.assemble_simulation_state

    def broken(*args, **kwargs):
        state = real(*args, **kwargs)
        # Remove a required attribute
        state.grid = None
        return state

    monkeypatch.setattr(mod, "assemble_simulation_state", broken)

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step1_state(copy.deepcopy(MINIMAL_VALID_INPUT))

    assert "Output schema validation FAILED" in str(excinfo.value)


# ------------------------------------------------------------
# Test 7 — Full happy path
# ------------------------------------------------------------
def test_step1_happy_path():
    state = orchestrate_step1_state(copy.deepcopy(MINIMAL_VALID_INPUT))

    # Basic structural checks
    assert state.grid["nx"] == 2
    assert state.fields["P"].shape == (2, 2, 2)
    assert state.mask.shape == (2, 2, 2)
    assert state.constants.dt > 0
