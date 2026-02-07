# tests/step_2/test_orchestrate_step2_health_structure.py

import numpy as np

from tests.helpers.schema_dummy_state import SchemaDummyState
from src.step2.orchestrate_step2 import orchestrate_step2


def test_orchestrate_step2_health_structure():
    # Create a fully Step‑1‑schema‑compliant dummy state
    state = SchemaDummyState(4, 4, 4)

    # Run Step 2 orchestrator
    result = orchestrate_step2(state)

    # Health block (lowercase key in new schema)
    health = result["health"]

    assert "initial_divergence_norm" in health
    assert "max_velocity_magnitude" in health
    assert "cfl_advection_estimate" in health
