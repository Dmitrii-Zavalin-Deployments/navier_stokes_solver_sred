# tests/step_2/test_orchestrate_step2_operators_present.py

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.dummy_state_step2 import DummyState


def test_orchestrate_step2_operators_present():
    state = DummyState(4, 4, 4)
    result = orchestrate_step2(state)

    ops = result["Operators"]

    expected = [
        "divergence",
        "gradient_p_x",
        "gradient_p_y",
        "gradient_p_z",
        "laplacian_u",
        "laplacian_v",
        "laplacian_w",
        "advection_u",
        "advection_v",
        "advection_w",
    ]

    for name in expected:
        assert name in ops, f"Missing operator: {name}"
        assert callable(ops[name]), f"Operator {name} is not callable"
