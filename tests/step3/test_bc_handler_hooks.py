# tests/step3/test_bc_handler_hooks.py

import numpy as np
from src.step3.orchestrate_step3 import step3

class DummyHandler:
    def __init__(self):
        self.pre_called = False
        self.post_called = False

    def apply_pre(self, state):
        self.pre_called = True

    def apply_post(self, state):
        self.post_called = True


def test_bc_handler_hooks_called(minimal_state):
    handler = DummyHandler()
    minimal_state["BC_handler"] = handler

    step3(minimal_state, current_time=0.0, step_index=0)

    assert handler.pre_called
    assert handler.post_called
