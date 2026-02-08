# tests/step1/test_state_to_dict.py

import numpy as np
from src.step1.construct_simulation_state import construct_simulation_state
from tests.helpers.minimal_step1_input import MINIMAL_VALID_INPUT


def test_construct_simulation_state_output_serializable():
    state = construct_simulation_state(MINIMAL_VALID_INPUT.copy())

    # Check top-level keys
    assert "config" in state
    assert "grid" in state
    assert "fields" in state
    assert "mask_3d" in state
    assert "constants" in state

    # Arrays must be JSON-serializable lists
    assert isinstance(state["fields"]["P"], list)
    assert isinstance(state["mask_3d"], list)

    # Nested arrays must also be lists
    assert isinstance(state["fields"]["U"], list)
    assert isinstance(state["fields"]["V"], list)
    assert isinstance(state["fields"]["W"], list)
