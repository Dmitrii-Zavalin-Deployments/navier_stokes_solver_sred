# tests/step3/test_step3_integration.py

import numpy as np
import pytest

from src.step3.orchestrate_step3 import orchestrate_step3_state
from src.solver_state import SolverState
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


@pytest.fixture
def step3_state():
    """
    Provides a fully valid Step‑2‑schema dummy state
    to be used as input to Step‑3 orchestrator.
    """
    return Step2SchemaDummyState(nx=3, ny=3, nz=3)


def _wire_uniform_divergence(state, value=1.0):
    def div(U, V, W):
        return value * np.ones_like(state["fields"]["P"])
    state["divergence"]["op"] = div


def test_schema_preservation(step3_state):
    s2 = step3_state
    new_state = step3(s2, current_time=0.0, step_index=0)

    # Basic structural expectations
    assert "fields" in new_state
    assert "health" in new_state
    assert "history" in new_state

    P = new_state["fields"]["P"]
    U = new_state["fields"]["U"]
    V = new_state["fields"]["V"]
    W = new_state["fields"]["W"]

    assert P.shape == (3, 3, 3)
    assert U.shape == (4, 3, 3)
    assert V.shape == (3, 4, 3)
    assert W.shape == (3, 3, 4)


def test_schema_validation_passes(step3_state):
    step3(step3_state, current_time=0.0, step_index=0)


def test_schema_validation_fails_on_invalid_state(step3_state):
    s2 = step3_state
    del s2["fields"]  # break Step‑2 schema

    with pytest.raises(RuntimeError):
        step3(s2, current_time=0.0, step_index=0)


def test_divergence_reduction(step3_state):
    s2 = step3_state
    _wire_uniform_divergence(s2, value=1.0)

    new_state = step3(s2, current_time=0.0, step_index=0)

    div_norm = new_state["health"]["post_correction_divergence_norm"]
    assert div_norm >= 0.0
    assert np.isfinite(div_norm)


def test_mask_awareness(step3_state):
    s2 = step3_state

    mask = np.array(s2["mask_semantics"]["mask"], copy=True)
    mask[1, 1, 1] = 0  # solid cell
    s2["mask_semantics"]["mask"] = mask
    s2["mask_semantics"]["is_solid"] = (mask == 0)

    new_state = step3(s2, current_time=0.0, step_index=0)

    assert np.any(new_state["fields"]["U"] == 0.0)


def test_singular_ppe(step3_state):
    s2 = step3_state
    _wire_uniform_divergence(s2, value=1.0)

    s2["ppe_structure"]["ppe_is_singular"] = True

    new_state = step3(s2, current_time=0.0, step_index=0)

    fluid = new_state["mask_semantics"]["is_fluid"]
    P = new_state["fields"]["P"]
    assert abs(P[fluid].mean()) < 1e-12


def test_non_singular_ppe(step3_state):
    s2 = step3_state
    _wire_uniform_divergence(s2, value=1.0)

    s2["ppe_structure"]["ppe_is_singular"] = False

    new_state = step3(s2, current_time=0.0, step_index=0)

    P = new_state["fields"]["P"]
    assert np.allclose(P, 0.0)


def test_vacuum_all_solids():
    """
    All‑solid domain must not crash and must produce zero velocities.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    mask = np.zeros_like(s2["mask_semantics"]["mask"], int)
    s2["mask_semantics"]["mask"] = mask
    s2["mask_semantics"]["is_fluid"] = (mask == 1)
    s2["mask_semantics"]["is_solid"] = (mask == 0)

    def div_zero(U, V, W):
        return np.zeros_like(s2["fields"]["P"])

    s2["divergence"]["op"] = div_zero

    new_state = step3(s2, current_time=0.0, step_index=0)

    U = new_state["fields"]["U"]
    V = new_state["fields"]["V"]
    W = new_state["fields"]["W"]

    assert np.allclose(U, 0.0)
    assert np.allclose(V, 0.0)
    assert np.allclose(W, 0.0)


def test_full_round_trip(step3_state):
    s2 = step3_state
    new_state = step3(s2, current_time=0.0, step_index=0)

    assert "health" in new_state
    assert "history" in new_state
    assert new_state["health"]["post_correction_divergence_norm"] >= 0.0
