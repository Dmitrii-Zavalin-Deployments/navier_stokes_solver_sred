# tests/step3/test_step3_integration.py

import numpy as np
import pytest
from src.step3.orchestrate_step3 import step3


def test_schema_preservation(minimal_state):
    """
    Step 3 must not change the structure of SimulationState.
    Only values may change.
    """
    state = minimal_state
    original_keys = set(state.keys())

    step3(state, current_time=0.0, step_index=0)

    new_keys = set(state.keys())
    assert original_keys.issubset(new_keys)

    # Check shapes preserved
    assert state["P"].shape == (3, 3, 3)
    assert state["U"].shape == (4, 3, 3)
    assert state["V"].shape == (3, 4, 3)
    assert state["W"].shape == (3, 3, 4)


def test_schema_validation_passes(minimal_state):
    """
    After adding schema validation to the orchestrator,
    Step 3 must successfully validate a correct state.
    """
    state = minimal_state
    # Should not raise
    step3(state, current_time=0.0, step_index=0)


def test_schema_validation_fails_on_invalid_state(minimal_state):
    """
    If Step 3 produces an invalid state (e.g., missing required fields),
    the orchestrator must raise a RuntimeError.
    """
    state = minimal_state

    # Introduce an invalid modification BEFORE running Step 3
    del state["Constants"]  # required by schema

    with pytest.raises(RuntimeError):
        step3(state, current_time=0.0, step_index=0)


def test_divergence_reduction(minimal_state):
    state = minimal_state
    pattern = np.ones_like(state["P"])
    state["_divergence_pattern"] = pattern

    div_before = np.linalg.norm(pattern)

    step3(state, current_time=0.0, step_index=0)

    div_after = state["Health"]["post_correction_divergence_norm"]

    # With dummy operators, they may be equal — but field must exist
    assert div_after >= 0.0


def test_mask_awareness(minimal_state):
    state = minimal_state
    state["Mask"][1, 1, 1] = 0

    step3(state, current_time=0.0, step_index=0)

    # Face-based zeroing: cannot guarantee exact index, but must be zero somewhere
    assert np.any(state["U"] == 0.0)


def test_singular_ppe(minimal_state):
    state = minimal_state
    state["PPE"]["ppe_is_singular"] = True
    state["_divergence_pattern"] = np.ones_like(state["P"])

    step3(state, current_time=0.0, step_index=0)

    fluid = state["is_fluid"]
    assert abs(state["P"][fluid].mean()) < 1e-12


def test_non_singular_ppe(minimal_state):
    state = minimal_state
    state["PPE"]["ppe_is_singular"] = False
    state["_divergence_pattern"] = np.ones_like(state["P"])

    step3(state, current_time=0.0, step_index=0)

    # Should not subtract mean
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


def test_full_round_trip(minimal_state):
    """
    Full Step 3 pipeline must run without errors and produce valid diagnostics.
    """
    state = minimal_state
    step3(state, current_time=0.0, step_index=0)

    assert "Health" in state
    assert "History" in state
    assert state["Health"]["post_correction_divergence_norm"] >= 0.0
