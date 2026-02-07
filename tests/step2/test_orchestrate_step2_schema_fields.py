# tests/step2/test_orchestrate_step2_schema_fields.py

import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2

from tests.helpers.schema_dummy_state import SchemaDummyState


def test_orchestrate_step2_schema_fields():
    # Create a fully Step‑1‑schema‑compliant dummy state
    state = SchemaDummyState(4, 4, 4)

    # Run orchestrator
    result = orchestrate_step2(state)

    required = [
        "constants",
        "fields",
        "is_fluid",
        "is_boundary_cell",
        "operators",
        "ppe",
        "health",
        "advection_meta",
    ]

    for key in required:
        assert key in result, f"Missing required field: {key}"
