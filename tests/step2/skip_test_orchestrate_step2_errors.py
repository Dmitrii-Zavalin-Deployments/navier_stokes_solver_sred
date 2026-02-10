# tests/step2/test_orchestrate_step2_errors.py

import pytest
import numpy as np

from src.step2.orchestrate_step2 import orchestrate_step2


def make_minimal_step1_state():
    return {
        "grid": {
            "nx": 4, "ny": 4, "nz": 4,
            "dx": 1.0, "dy": 1.0, "dz": 1.0,
        },
        "config": {
            "fluid": {"density": 1.0, "viscosity": 0.1},
            "simulation": {"dt": 0.1, "advection_scheme": "central"},
        },
        "fields": {
            "U": np.zeros((5, 4, 4)).tolist(),
            "V": np.zeros((4, 5, 4)).tolist(),
            "W": np.zeros((4, 4, 5)).tolist(),
            "P": np.zeros((4, 4, 4)).tolist(),
        },
        "mask_3d": np.ones((4, 4, 4), int).tolist(),
        "boundary_table_list": [],
    }


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

    required_keys = {
        "constants",
        "mask_semantics",
        "fluid_mask",
        "divergence",
        "pressure_gradients",
        "laplacians",
        "advection",
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
    state["boundary_table_list"] = []

    out = orchestrate_step2(state)

    assert "ppe_structure" in out
    assert out["ppe_structure"]["ppe_is_singular"] is True
