# tests/step1/test_step1_output_dummy_schema.py

import numpy as np

from tests.helpers.solver_step1_output_dummy import make_step1_dummy_state
from tests.helpers.solver_step1_output_schema import STEP1_OUTPUT_SCHEMA


def test_step1_dummy_matches_schema():
    state = make_step1_dummy_state()

    # Top-level attributes
    for attr in STEP1_OUTPUT_SCHEMA.keys():
        assert hasattr(state, attr), f"Missing attribute: state.{attr}"

    # Grid
    for key, expected_type in STEP1_OUTPUT_SCHEMA["grid"].items():
        assert hasattr(state.grid, key), f"Missing grid.{key}"
        assert isinstance(getattr(state.grid, key), expected_type)

    # Config
    for key, expected_type in STEP1_OUTPUT_SCHEMA["config"].items():
        assert hasattr(state.config, key), f"Missing config.{key}"
        assert isinstance(getattr(state.config, key), expected_type)

    # Constants
    for key, expected_type in STEP1_OUTPUT_SCHEMA["constants"].items():
        assert key in state.constants
        assert isinstance(state.constants[key], expected_type)

    # Mask
    assert isinstance(state.mask, np.ndarray)

    # Fields
    for field_name in STEP1_OUTPUT_SCHEMA["fields"].keys():
        assert field_name in state.fields
        assert isinstance(state.fields[field_name], np.ndarray)

    # Boundary conditions
    assert isinstance(state.boundary_conditions, dict)

    # Health
    assert isinstance(state.health, dict)
