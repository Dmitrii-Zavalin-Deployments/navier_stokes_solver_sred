# tests/step1/test_step1_output_dummy_schema.py

import numpy as np
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step1_output_schema import EXPECTED_STEP1_SCHEMA


def test_step1_dummy_matches_schema():
    state = make_step1_output_dummy()

    # Top-level keys
    for key in EXPECTED_STEP1_SCHEMA:
        assert hasattr(state, key), f"Missing key: {key}"

    # Fields
    for f in ["P", "U", "V", "W"]:
        assert f in state.fields
        assert isinstance(state.fields[f], np.ndarray)

    # Mask semantics
    assert isinstance(state.mask, np.ndarray)
    assert isinstance(state.is_fluid, np.ndarray)
    assert isinstance(state.is_boundary_cell, np.ndarray)

    # Constants
    assert isinstance(state.constants, dict)

    # Boundary conditions
    assert state.boundary_conditions is None or callable(state.boundary_conditions)

    # Operators, PPE, health, history must exist (empty)
    assert isinstance(state.operators, dict)
    assert isinstance(state.ppe, dict)
    assert isinstance(state.health, dict)
    assert isinstance(state.history, dict)
