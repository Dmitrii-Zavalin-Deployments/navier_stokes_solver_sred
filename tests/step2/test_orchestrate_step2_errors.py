# tests/step2/test_orchestrate_step2_errors.py

import pytest
import numpy as np

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.step1_schema_dummy_state import Step1SchemaDummyState


def make_minimal_step1_state():
    """
    Use the canonical, fully Step‑1‑schema‑compliant dummy.
    Ensures Step‑2 receives a valid Step‑1 output.
    """
    return Step1SchemaDummyState(nx=4, ny=4, nz=4)


# ------------------------------------------------------------
# Test 1 — orchestrator rejects malformed Step‑1 input
# ------------------------------------------------------------
def test_orchestrate_step2_rejects_missing_grid():
    state = make_minimal_step1_state()
    state.pop("grid")

    with pytest.raises(KeyError):
        orchestrate_step2(state)


def test_orchestrate_step2_rejects_missing_config():
    state = make_minimal_step1_state()
    state.pop("config")

    with pytest.raises(KeyError):
        orchestrate_step2(state)


def test_orchestrate_step2_rejects_missing_fields():
    state = make_minimal_step1_state()
    state.pop("fields")

    with pytest.raises(KeyError):
        orchestrate_step2(state)


def test_orchestrate_step2_rejects_missing_mask():
    state = make_minimal_step1_state()
    state.pop("mask_3d")

    with pytest.raises(KeyError):
        orchestrate_step2(state)


# ------------------------------------------------------------
# Test 2 — orchestrator returns a valid Step‑2 output dict
# ------------------------------------------------------------
def test_orchestrate_step2_output_structure():
    state = make_minimal_step1_state()
    out = orchestrate_step2(state)

    # Updated to match the actual Step‑2 output schema
    required_keys = {
        "grid",
        "fields",
        "config",
        "constants",
        "mask",
        "is_fluid",
        "is_solid",
        "is_boundary_cell",
        "operators",
        "ppe",
        "ppe_structure",
        "health",
        "meta",
    }

    assert required_keys.issubset(out.keys())


# ------------------------------------------------------------
# Test 3 — orchestrator does not mutate input state
# ------------------------------------------------------------
def test_orchestrate_step2_does_not_mutate_input():
    state = make_minimal_step1_state()
    original = repr(state)

    orchestrate_step2(state)

    assert repr(state) == original


# ------------------------------------------------------------
# Test 4 — orchestrator handles empty boundary table
# ------------------------------------------------------------
def test_orchestrate_step2_empty_boundary_table():
    state = make_minimal_step1_state()

    # Replace boundary table with empty lists (still schema‑valid)
    state["boundary_table"] = {
        "x_min": [],
        "x_max": [],
        "y_min": [],
        "y_max": [],
        "z_min": [],
        "z_max": [],
    }

    out = orchestrate_step2(state)

    assert "ppe_structure" in out
    assert out["ppe_structure"]["ppe_is_singular"] is True
