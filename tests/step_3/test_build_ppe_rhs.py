# tests/step3/test_build_ppe_rhs.py

import numpy as np
from src.step3.build_ppe_rhs import build_ppe_rhs

def test_zero_divergence(minimal_state):
    state = minimal_state
    rhs = build_ppe_rhs(state, state["U"], state["V"], state["W"])
    assert np.allclose(rhs, 0.0)

def test_uniform_divergence(minimal_state):
    state = minimal_state
    pattern = np.ones_like(state["P"])
    state["_divergence_pattern"] = pattern
    rhs = build_ppe_rhs(state, state["U"], state["V"], state["W"])
    rho = state["Constants"]["rho"]
    dt = state["Constants"]["dt"]
    assert np.allclose(rhs, rho/dt)

def test_solid_zeroing(minimal_state):
    state = minimal_state
    state["_divergence_pattern"] = np.ones_like(state["P"])
    state["Mask"][1,1,1] = 0
    rhs = build_ppe_rhs(state, state["U"], state["V"], state["W"])
    assert rhs[1,1,1] == 0.0

def test_minimal_grid():
    state = {
        "Mask": np.ones((1,1,1), int),
        "is_fluid": np.ones((1,1,1), bool),
        "P": np.zeros((1,1,1)),
        "Constants": {"rho":1,"dt":0.1},
        "Operators": {"divergence": lambda U,V,W,s: np.zeros((1,1,1))}
    }
    rhs = build_ppe_rhs(state, np.zeros((2,1,1)), np.zeros((1,2,1)), np.zeros((1,1,2)))
    assert rhs.shape == (1,1,1)
