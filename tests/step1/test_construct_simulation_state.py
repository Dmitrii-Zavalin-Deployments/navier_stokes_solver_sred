# tests/step1/test_orchestrate_step1.py

import pytest
import copy

from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.minimal_step1_input import MINIMAL_VALID_INPUT


# ------------------------------------------------------------
# Test 1 — Input schema validation failure
# ------------------------------------------------------------
def test_step1_input_schema_failure():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad.pop("domain_definition")  # required key

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step1(bad)

    assert "Input schema validation FAILED" in str(excinfo.value)


# ------------------------------------------------------------
# Test 2 — Physical constraints failure
# ------------------------------------------------------------
def test_step1_physical_constraints_failure():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["fluid_properties"]["density"] = -1.0  # invalid

    with pytest.raises(Exception):
        orchestrate_step1(bad)


# ------------------------------------------------------------
# Test 3 — Geometry mask mapping
# ------------------------------------------------------------
def test_step1_geometry_mask_mapping():
    state = orchestrate_step1(copy.deepcopy(MINIMAL_VALID_INPUT))

    # mask_3d must be 2×2×2
    assert state["mask_3d"].shape == (2, 2, 2)

    # all values must be 1 (fluid)
    assert (state["mask_3d"] == 1).all()


# ------------------------------------------------------------
# Test 4 — Boundary conditions normalization
# ------------------------------------------------------------
def test_step1_boundary_conditions():
    state = orchestrate_step1(copy.deepcopy(MINIMAL_VALID_INPUT))

    # boundary_table must exist and be non-empty
    assert "boundary_table" in state
    assert len(state["boundary_table"]) >= 1


# ------------------------------------------------------------
# Test 5 — Derived constants correctness
# ------------------------------------------------------------
def test_step1_derived_constants():
    state = orchestrate_step1(copy.deepcopy(MINIMAL_VALID_INPUT))

    c = state["constants"]

    assert c["rho"] == 1.0
    assert c["mu"] == 0.1
    assert c["dt"] > 0
    assert c["inv_dx"] == 1.0 / c["dx"]


# ------------------------------------------------------------
# Test 6 — verify_staggered_shapes failure
# ------------------------------------------------------------
def test_step1_shape_verification_failure(monkeypatch):
    import src.step1.orchestrate_step1 as mod

    # Monkeypatch verify_staggered_shapes to force failure
    def fail(_):
        raise ValueError("shape mismatch")

    monkeypatch.setattr(mod, "verify_staggered_shapes", fail)

    with pytest.raises(ValueError):
        orchestrate_step1(copy.deepcopy(MINIMAL_VALID_INPUT))


# ------------------------------------------------------------
# Test 7 — Output schema validation failure
# ------------------------------------------------------------
def test_step1_output_schema_failure(monkeypatch):
    import src.step1.orchestrate_step1 as mod

    # Monkeypatch assemble_simulation_state to break output schema
    real = mod.assemble_simulation_state

    def broken(*args, **kwargs):
        d = real(*args, **kwargs)
        d.pop("grid")  # required by output schema
        return d

    monkeypatch.setattr(mod, "assemble_simulation_state", broken)

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step1(copy.deepcopy(MINIMAL_VALID_INPUT))

    assert "Output schema validation FAILED" in str(excinfo.value)


# ------------------------------------------------------------
# Test 8 — Full happy path
# ------------------------------------------------------------
def test_step1_happy_path():
    state = orchestrate_step1(copy.deepcopy(MINIMAL_VALID_INPUT))

    # Basic structural checks
    assert state["grid"]["nx"] == 2
    assert state["fields"]["P"].shape == (2, 2, 2)
    assert state["mask_3d"].shape == (2, 2, 2)
    assert state["constants"]["dt"] > 0
