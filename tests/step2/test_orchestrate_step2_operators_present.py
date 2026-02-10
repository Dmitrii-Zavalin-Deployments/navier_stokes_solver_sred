# tests/step2/test_orchestrate_step2_operators_present.py

import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.step1_schema_dummy_state import Step1SchemaDummyState


def make_minimal_step1_state():
    """
    Use the canonical, fully Step‑1‑schema‑compliant dummy.
    Ensures Step‑2 receives a valid Step‑1 output.
    """
    return Step1SchemaDummyState(nx=4, ny=4, nz=4)


def test_orchestrate_step2_operators_present():
    # Create a fully Step‑1‑schema‑compliant dummy state
    state = make_minimal_step1_state()

    # Run orchestrator
    result = orchestrate_step2(state)

    # Operators block (schema requires STRINGS)
    ops = result["operators"]

    expected = [
        "divergence",
        "gradient_p_x",
        "gradient_p_y",
        "gradient_p_z",
        "laplacian_u",
        "laplacian_v",
        "laplacian_w",
        "advection_u",
        "advection_v",
        "advection_w",
    ]

    for name in expected:
        assert name in ops, f"Missing operator: {name}"
        assert isinstance(ops[name], str), f"Operator {name} must be a string"
