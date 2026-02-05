# tests/step_2/test_orchestrate_step2_ppe_structure.py

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.dummy_state_step2 import DummyState


def test_orchestrate_step2_ppe_structure():
    state = DummyState(4, 4, 4)
    result = orchestrate_step2(state)

    ppe = result["PPE"]

    assert "rhs_builder" in ppe and callable(ppe["rhs_builder"])
    assert "solver_type" in ppe
    assert "tolerance" in ppe
    assert "max_iterations" in ppe
    assert "ppe_is_singular" in ppe
