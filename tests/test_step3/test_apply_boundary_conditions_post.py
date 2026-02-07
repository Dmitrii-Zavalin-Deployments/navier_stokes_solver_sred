# tests/step3/test_apply_boundary_conditions_post.py

import numpy as np
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post

def test_state_update(minimal_state):
    state = minimal_state
    U_new = np.ones_like(state["U"])
    V_new = np.ones_like(state["V"])
    W_new = np.ones_like(state["W"])
    P_new = np.ones_like(state["P"])

    apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new)

    assert np.allclose(state["U"], U_new)
    assert np.allclose(state["P"], P_new)

def test_bc_handler(minimal_state):
    class Handler:
        def __init__(self): self.called = False
        def apply_post(self, state): self.called = True

    state = minimal_state
    handler = Handler()
    state["BC_handler"] = handler

    apply_boundary_conditions_post(state, state["U"], state["V"], state["W"], state["P"])
    assert handler.called

def test_minimal_grid():
    state = {
        "Mask": np.ones((1,1,1), int),
        "U": np.zeros((2,1,1)),
        "V": np.zeros((1,2,1)),
        "W": np.zeros((1,1,2)),
        "P": np.zeros((1,1,1)),
    }
    apply_boundary_conditions_post(state, state["U"], state["V"], state["W"], state["P"])
