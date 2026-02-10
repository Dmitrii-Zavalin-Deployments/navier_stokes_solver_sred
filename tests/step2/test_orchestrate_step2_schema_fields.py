# tests/step2/test_orchestrate_step2_schema_fields.py

import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.step1_schema_dummy_state import Step1SchemaDummyState


def make_minimal_step1_state():
    """
    Use the canonical, fully schema‑compliant Step‑1 dummy.
    This ensures Step‑2 receives a valid Step‑1 output.
    """
    return Step1SchemaDummyState(nx=4, ny=4, nz=4)


def test_orchestrate_step2_schema_fields():
    # Create a fully Step‑1‑schema‑compliant dummy state
    state = make_minimal_step1_state()

    # Run orchestrator
    result = orchestrate_step2(state)

    # Required Step‑2 output fields (schema‑aligned)
    required = [
        "grid",
        "fields",
        "config",
        "constants",
        "mask",
        "is_fluid",
        "is_solid",
        "is_boundary_cell",
        "operators",
        "ppe",
        "ppe_structure",
        "health",
        "meta",
    ]

    for key in required:
        assert key in result, f"Missing required field: {key}"
