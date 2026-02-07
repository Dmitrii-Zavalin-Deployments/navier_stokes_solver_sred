# tests/step2/test_orchestrate_step2_operators_present.py

import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2

from tests.helpers.schema_dummy_state import SchemaDummyState


def test_orchestrate_step2_operators_present():
    # Create a fully Step‑1‑schema‑compliant dummy state
    state = SchemaDummyState(4, 4, 4)

    # Run orchestrator
    result = orchestrate_step2(state)

    # NEW lowercase key
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
        assert callable(ops[name]), f"Operator {name} is not callable"
