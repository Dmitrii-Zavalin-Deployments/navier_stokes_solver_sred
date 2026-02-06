# tests/step3/test_apply_boundary_conditions_pre.py

import numpy as np
from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre

def test_solid_zeroing(minimal_state):
    state = minimal_state
    state["Mask"][1,1,1] = 0
    state["U"].fill(1.0)
    apply_boundary_conditions_pre(state)
    assert np.any(state["U"] == 0.0)

def test_bc_handler_invocation(minimal_state):
    class Handler:
        def __init__(self): self.called = False
        def apply_pre(self, state): self.called = True

    state = minimal_state
    handler = Handler()
    state["BC_handler"] = handler

    apply_boundary_conditions_pre(state)
    assert handler.called

def test_pressure_shape_preserved(minimal_state):
    state = minimal_state
    P_before = state["P"].copy()
    apply_boundary_conditions_pre(state)
    assert state["P"].shape == P_before.shape

def test_no_bc_handler(minimal_state):
    state = minimal_state
    apply_boundary_conditions_pre(state)  # should not crash

def test_minimal_grid():
    state = {
        "Mask": np.ones((1,1,1), int),
        "U": np.zeros((2,1,1)),
        "V": np.zeros((1,2,1)),
        "W": np.zeros((1,1,2)),
        "P": np.zeros((1,1,1)),
        "is_fluid": np.ones((1,1,1), bool),
        "is_boundary_cell": np.zeros((1,1,1), bool),
        "BCs": [],
    }
    apply_boundary_conditions_pre(state)
