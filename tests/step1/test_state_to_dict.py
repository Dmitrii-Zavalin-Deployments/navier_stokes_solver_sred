# tests/step1/test_state_to_dict.py

import numpy as np
from src.step1.construct_simulation_state import construct_simulation_state
from src.step1.state_to_dict import state_to_dict
from tests.helpers.minimal_step1_input import MINIMAL_VALID_INPUT


def test_construct_simulation_state_output_serializable():
    # Step‑1 returns NumPy arrays (runtime state)
    state = construct_simulation_state(MINIMAL_VALID_INPUT.copy())

    # Convert to JSON‑safe lists for serialization tests
    state_dict = state_to_dict(state)

    # Check top-level keys
    assert "config" in state_dict
    assert "grid" in state_dict
    assert "fields" in state_dict
    assert "mask_3d" in state_dict
    assert "constants" in state_dict

    # Arrays must be JSON-serializable lists
    assert isinstance(state_dict["fields"]["P"], list)
    assert isinstance(state_dict["mask_3d"], list)

    # Nested arrays must also be lists
    assert isinstance(state_dict["fields"]["U"], list)
    assert isinstance(state_dict["fields"]["V"], list)
    assert isinstance(state_dict["fields"]["W"], list)
