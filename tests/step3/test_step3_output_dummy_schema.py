# tests/step3/test_step3_output_dummy_schema.py

from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step3_output_schema import EXPECTED_STEP3_SCHEMA
import numpy as np


def test_step3_dummy_matches_schema():
    state = make_step3_output_dummy()

    # Top-level keys
    for key in EXPECTED_STEP3_SCHEMA:
        assert hasattr(state, key), f"Missing key: {key}"

    # Fields
    for f in ["U", "V", "W", "P"]:
        assert f in state.fields
        assert isinstance(state.fields[f], np.ndarray)

    # Health
    health_schema = EXPECTED_STEP3_SCHEMA["health"]
    for key, typ in health_schema.items():
        assert key in state.health
        assert isinstance(state.health[key], typ)

    # History
    hist_schema = EXPECTED_STEP3_SCHEMA["history"]
    for key, typ in hist_schema.items():
        assert key in state.history
        assert isinstance(state.history[key], typ)
