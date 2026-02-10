# tests/step2/test_orchestrate_step2_no_validator.py

import numpy as np
import src.step2.orchestrate_step2 as o
from tests.helpers.step1_schema_dummy_state import Step1SchemaDummyState


def make_minimal_step1_state():
    """
    Use the canonical, fully Step‑1‑schema‑compliant dummy.
    Ensures Step‑2 receives a valid Step‑1 output even when
    schema validation is disabled.
    """
    return Step1SchemaDummyState(nx=4, ny=4, nz=4)


def test_orchestrate_step2_no_validator():
    # Simulate missing validator
    o.validate_json_schema = None
    o.load_schema = None

    # Create a fully Step‑1‑schema‑compliant dummy state
    state = make_minimal_step1_state()

    # Run orchestrator
    result = o.orchestrate_step2(state)

    # NEW: Step 2 stores constants under lowercase "constants"
    assert "constants" in result
