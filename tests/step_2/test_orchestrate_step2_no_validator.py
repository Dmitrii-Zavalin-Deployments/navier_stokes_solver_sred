# tests/step_2/test_orchestrate_step2_no_validator.py

import src.step2.orchestrate_step2 as o
from tests.helpers.dummy_state_step2 import DummyState


def test_orchestrate_step2_no_validator():
    # Simulate missing validator
    o.validate_json_schema = None

    state = DummyState(4, 4, 4)
    result = o.orchestrate_step2(state)

    assert "Constants" in result
