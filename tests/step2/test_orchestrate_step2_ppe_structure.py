# tests/step2/test_orchestrate_step2_ppe_structure.py

import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2

from tests.helpers.schema_dummy_state import SchemaDummyState


def test_orchestrate_step2_ppe_structure():
    # Create a fully Step‑1‑schema‑compliant dummy state
    state = SchemaDummyState(4, 4, 4)

    # Run orchestrator
    result = orchestrate_step2(state)

    # NEW lowercase key
    ppe = result["ppe"]

    assert "rhs_builder" in ppe and callable(ppe["rhs_builder"])
    assert "solver_type" in ppe
    assert "tolerance" in ppe
    assert "max_iterations" in ppe
    assert "ppe_is_singular" in ppe
