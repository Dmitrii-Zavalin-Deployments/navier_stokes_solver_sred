# tests/step2/test_orchestrate_step2_ppe_structure.py

import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.step1_schema_dummy_state import Step1SchemaDummyState


def make_minimal_step1_state():
    """
    Use the canonical, fully Step‑1‑schema‑compliant dummy.
    Ensures Step‑2 receives a valid Step‑1 output.
    """
    return Step1SchemaDummyState(nx=4, ny=4, nz=4)


def test_orchestrate_step2_ppe_structure():
    # Create a fully Step‑1‑schema‑compliant dummy state
    state = make_minimal_step1_state()

    # Run orchestrator
    result = orchestrate_step2(state)

    # PPE block (schema requires STRINGS, not callables)
    ppe = result["ppe"]

    assert "rhs_builder" in ppe
    assert isinstance(ppe["rhs_builder"], str)

    assert "solver_type" in ppe
    assert isinstance(ppe["solver_type"], str)

    assert "tolerance" in ppe
    assert isinstance(ppe["tolerance"], (int, float))

    assert "max_iterations" in ppe
    assert isinstance(ppe["max_iterations"], int)

    assert "ppe_is_singular" in ppe
    assert isinstance(ppe["ppe_is_singular"], bool)
