# tests/step1/test_state_to_dict.py

import numpy as np
from src.step1.construct_simulation_state import _state_to_dict
from tests.helpers.step1_schema_dummy_state import Step1SchemaDummyState


def test_state_to_dict_conversion():
    state = Step1SchemaDummyState(2, 2, 2)
    d = _state_to_dict(state)

    # Check top-level keys
    assert "config" in d
    assert "grid" in d
    assert "fields" in d
    assert "mask_3d" in d
    assert "constants" in d

    # Check arrays converted to lists
    assert isinstance(d["fields"]["P"], list)
    assert isinstance(d["mask_3d"], list)
