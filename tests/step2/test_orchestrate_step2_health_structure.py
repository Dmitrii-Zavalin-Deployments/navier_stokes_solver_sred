# tests/step2/test_orchestrate_step2_health_structure.py

import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.step1_schema_dummy_state import Step1SchemaDummyState


def make_minimal_step1_state():
    """
    Use the canonical, fully Step‑1‑schema‑compliant dummy.
    Ensures Step‑2 receives a valid Step‑1 output.
    """
    return Step1SchemaDummyState(nx=4, ny=4, nz=4)


def test_orchestrate_step2_health_structure():
    # Create a fully Step‑1‑schema‑compliant dummy state
    state = make_minimal_step1_state()

    # Run Step 2 orchestrator
    result = orchestrate_step2(state)

    # Health block (lowercase key in new schema)
    health = result["health"]

    assert "initial_divergence_norm" in health
    assert "max_velocity_magnitude" in health
    assert "cfl_advection_estimate" in health
