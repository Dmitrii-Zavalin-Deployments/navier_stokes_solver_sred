# tests/step_2/test_orchestrate_step2_health_structure.py

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.dummy_state_step2 import DummyState


def test_orchestrate_step2_health_structure():
    state = DummyState(4, 4, 4)
    result = orchestrate_step2(state)

    health = result["Health"]

    assert "initial_divergence_norm" in health
    assert "max_velocity_magnitude" in health
    assert "cfl_advection_estimate" in health
