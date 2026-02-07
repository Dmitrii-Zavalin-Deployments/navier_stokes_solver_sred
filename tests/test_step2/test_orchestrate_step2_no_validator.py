# tests/step2/test_orchestrate_step2_no_validator.py

import numpy as np
import src.step2.orchestrate_step2 as o

from tests.helpers.schema_dummy_state import SchemaDummyState


def test_orchestrate_step2_no_validator():
    # Simulate missing validator
    o.validate_json_schema = None

    # Create a fully Step‑1‑schema‑compliant dummy state
    state = SchemaDummyState(4, 4, 4)

    # Run orchestrator
    result = o.orchestrate_step2(state)

    # NEW: Step 2 stores constants under lowercase "constants"
    assert "constants" in result
