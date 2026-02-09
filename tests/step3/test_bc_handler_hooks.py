# tests/step3/test_bc_handler_hooks_called.py

import numpy as np
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


def test_bc_handler_hooks_called():
    """
    Step‑3 must call:
        • state["boundary_conditions_pre"]
        • state["boundary_conditions_post"]
    if they are provided as callables.
    """

    # Create a valid Step‑2 dummy
    s2 = Step2SchemaDummyState(nx=4, ny=4, nz=4)

    calls = {"pre": False, "post": False}

    def bc_pre(state, fields):
        calls["pre"] = True
        return fields

    def bc_post(state, fields):
        calls["post"] = True
        return fields

    # Attach hooks
    s2["boundary_conditions_pre"] = bc_pre
    s2["boundary_conditions_post"] = bc_post

    # Run Step‑3
    new_state = orchestrate_step3(
        state=s2,
        current_time=0.0,
        step_index=0,
    )

    assert calls["pre"]
    assert calls["post"]
