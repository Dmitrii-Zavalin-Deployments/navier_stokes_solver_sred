# tests/step3/test_solve_pressure.py

import numpy as np
from src.step3.solve_pressure import solve_pressure

def test_shape_consistency(minimal_state):
    state = minimal_state
    rhs = np.random.randn(*state["P"].shape)
    P_new = solve_pressure(state, rhs)
    assert P_new.shape == rhs.shape

def test_singular_mean_subtraction(minimal_state):
    state = minimal_state
    state["PPE"]["ppe_is_singular"] = True
    rhs = np.ones_like(state["P"])
    P_new = solve_pressure(state, rhs)
    assert abs(P_new[state["is_fluid"]].mean()) < 1e-12

def test_non_singular(minimal_state):
    state = minimal_state
    state["PPE"]["ppe_is_singular"] = False
    rhs = np.ones_like(state["P"])
    P_new = solve_pressure(state, rhs)
    assert np.allclose(P_new, 0.0)

def test_convergence_flag(minimal_state):
    state = minimal_state
    rhs = np.zeros_like(state["P"])
    solve_pressure(state, rhs)
    assert "ppe_converged" in state["PPE"]

def test_minimal_grid():
    state = {
        "Mask": np.ones((1,1,1), int),
        "is_fluid": np.ones((1,1,1), bool),
        "P": np.zeros((1,1,1)),
        "PPE": {"solver": None, "ppe_is_singular": False},
    }
    rhs = np.zeros((1,1,1))
    solve_pressure(state, rhs)
