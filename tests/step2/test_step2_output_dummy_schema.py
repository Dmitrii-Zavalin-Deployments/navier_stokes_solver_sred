# tests/step2/test_step2_output_dummy_schema.py

import numpy as np
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step2_output_schema import EXPECTED_STEP2_SCHEMA


def test_step2_dummy_matches_schema():
    state = make_step2_output_dummy()

    # Top-level keys
    for key in EXPECTED_STEP2_SCHEMA:
        assert hasattr(state, key), f"Missing key: {key}"

    # Fields
    for f in ["P", "U", "V", "W"]:
        assert f in state.fields
        assert isinstance(state.fields[f], np.ndarray)

    # Mask semantics
    assert isinstance(state.mask, np.ndarray)
    assert isinstance(state.is_fluid, np.ndarray)
    assert isinstance(state.is_boundary_cell, np.ndarray)

    # Operators
    assert isinstance(state.operators, dict)
    for op in ["divergence", "grad_x", "grad_y", "grad_z", "lap_u", "lap_v", "lap_w"]:
        assert op in state.operators
        assert callable(state.operators[op])

    # PPE
    assert isinstance(state.ppe, dict)
    for key in ["solver_type", "tolerance", "max_iterations", "ppe_is_singular", "rhs_builder"]:
        assert key in state.ppe

    # Health
    assert isinstance(state.health, dict)
    for key in ["divergence_norm", "max_velocity", "cfl"]:
        assert key in state.health

    # History
    assert isinstance(state.history, dict)
