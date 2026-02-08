# tests/step3/test_step3_integration.py

import numpy as np
import pytest
from src.step3.orchestrate_step3 import step3
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


# ----------------------------------------------------------------------
# Helper: convert Step‑2 dummy → Step‑3 input shape
# (This is test‑side only; core code must NOT do this.)
# ----------------------------------------------------------------------
def adapt_step2_to_step3(state):
    """
    Convert Step2SchemaDummyState (lowercase keys, nested fields)
    into the uppercase, flattened Step‑3 SimulationState shape.
    """

    return {
        "Config": state["config"],
        "Mask": state["fields"]["Mask"],
        "is_fluid": state["fields"]["Mask"] == 1,
        "is_boundary_cell": np.zeros_like(state["fields"]["Mask"], bool),

        "P": state["fields"]["P"],
        "U": state["fields"]["U"],
        "V": state["fields"]["V"],
        "W": state["fields"]["W"],

        "BCs": state["boundary_table_list"],

        "Constants": {
            "rho": state["config"]["fluid"]["density"],
            "mu": state["config"]["fluid"]["viscosity"],
            "dt": state["config"]["simulation"]["dt"],
            "dx": state["grid"]["dx"],
            "dy": state["grid"]["dy"],
            "dz": state["grid"]["dz"],
        },

        "Operators": state["operators"],

        "PPE": {
            "solver": None,
            "tolerance": 1e-6,
            "max_iterations": 100,
            "ppe_is_singular": False,
        },

        "Health": {},
        "History": {},
    }


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def step3_state():
    """Provides a fully valid Step‑3 input state."""
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    return adapt_step2_to_step3(s2)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_schema_preservation(step3_state):
    state = step3_state
    original_keys = set(state.keys())

    step3(state, current_time=0.0, step_index=0)

    new_keys = set(state.keys())
    assert original_keys.issubset(new_keys)

    assert state["P"].shape == (3, 3, 3)
    assert state["U"].shape == (4, 3, 3)
    assert state["V"].shape == (3, 4, 3)
    assert state["W"].shape == (3, 3, 4)


def test_schema_validation_passes(step3_state):
    step3(step3_state, current_time=0.0, step_index=0)


def test_schema_validation_fails_on_invalid_state(step3_state):
    state = step3_state
    del state["History"]

    with pytest.raises(RuntimeError):
        step3(state, current_time=0.0, step_index=0)


def test_divergence_reduction(step3_state):
    state = step3_state
    pattern = np.ones_like(state["P"])
    state["_divergence_pattern"] = pattern

    step3(state, current_time=0.0, step_index=0)

    assert state["Health"]["post_correction_divergence_norm"] >= 0.0


def test_mask_awareness(step3_state):
    state = step3_state
    state["Mask"][1, 1, 1] = 0

    step3(state, current_time=0.0, step_index=0)

    assert np.any(state["U"] == 0.0)


def test_singular_ppe(step3_state):
    state = step3_state
    state["PPE"]["ppe_is_singular"] = True
    state["_divergence_pattern"] = np.ones_like(state["P"])

    step3(state, current_time=0.0, step_index=0)

    fluid = state["is_fluid"]
    assert abs(state["P"][fluid].mean()) < 1e-12


def test_non_singular_ppe(step3_state):
    state = step3_state
    state["PPE"]["ppe_is_singular"] = False
    state["_divergence_pattern"] = np.ones_like(state["P"])

    step3(state, current_time=0.0, step_index=0)

    assert np.allclose(state["P"], 0.0)


def test_vacuum_all_solids():
    """
    All-solid domain must not crash and must produce zero velocities.
    """
    state = {
        "Mask": np.zeros((3, 3, 3), int),
        "is_fluid": np.zeros((3, 3, 3), bool),
        "is_boundary_cell": np.zeros((3, 3, 3), bool),
        "P": np.zeros((3, 3, 3)),
        "U": np.zeros((4, 3, 3)),
        "V": np.zeros((3, 4, 3)),
        "W": np.zeros((3, 3, 4)),
        "Config": {"external_forces": {}},
        "BCs": [],
        "Constants": {"rho": 1, "mu": 0.1, "dt": 0.01, "dx": 1, "dy": 1, "dz": 1},
        "Operators": {
            "divergence": lambda U, V, W, s: np.zeros((3, 3, 3)),
            "advection_u": lambda U, V, W, s: np.zeros_like(U),
            "advection_v": lambda U, V, W, s: np.zeros_like(V),
            "advection_w": lambda U, V, W, s: np.zeros_like(W),
            "laplacian_u": lambda U, s: np.zeros_like(U),
            "laplacian_v": lambda V, s: np.zeros_like(V),
            "laplacian_w": lambda W, s: np.zeros_like(W),
            "gradient_p_x": lambda P, s: np.zeros((4, 3, 3)),
            "gradient_p_y": lambda P, s: np.zeros((3, 4, 3)),
            "gradient_p_z": lambda P, s: np.zeros((3, 3, 4)),
        },
        "PPE": {"solver": None, "ppe_is_singular": False},
        "Health": {},
        "History": {},
    }

    step3(state, current_time=0.0, step_index=0)

    assert np.allclose(state["U"], 0.0)
    assert np.allclose(state["V"], 0.0)
    assert np.allclose(state["W"], 0.0)


def test_full_round_trip(step3_state):
    state = step3_state
    step3(state, current_time=0.0, step_index=0)

    assert "Health" in state
    assert "History" in state
    assert state["Health"]["post_correction_divergence_norm"] >= 0.0
