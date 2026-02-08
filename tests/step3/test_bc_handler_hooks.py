import numpy as np
from src.step3.orchestrate_step3 import step3
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


class DummyHandler:
    def __init__(self):
        self.pre_called = False
        self.post_called = False

    def apply_pre(self, state):
        self.pre_called = True

    def apply_post(self, state):
        self.post_called = True


def test_bc_handler_hooks_called():
    # Create a Step‑2‑schema‑valid dummy state
    state = Step2SchemaDummyState(nx=4, ny=4, nz=4)

    # Attach the handler
    handler = DummyHandler()
    state["BC_handler"] = handler

    # Run Step 3
    step3(state, current_time=0.0, step_index=0)

    # Verify hooks were called
    assert handler.pre_called
    assert handler.post_called