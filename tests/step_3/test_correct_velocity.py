# tests/step3/test_correct_velocity.py

import numpy as np
from src.step3.correct_velocity import correct_velocity

def test_zero_gradient(minimal_state):
    state = minimal_state
    U_star = np.ones_like(state["U"])
    V_star = np.ones_like(state["V"])
    W_star = np.ones_like(state["W"])
    P_new = np.zeros_like(state["P"])
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_new)
    assert np.allclose(U_new, U_star)
    assert np.allclose(V_new, V_star)
    assert np.allclose(W_new, W_star)

def test_solid_mask(minimal_state):
    state = minimal_state
    state["Mask"][1,1,1] = 0
    U_star = np.ones_like(state["U"])
    V_star = np.ones_like(state["V"])
    W_star = np.ones_like(state["W"])
    P_new = np.zeros_like(state["P"])
    U_new, _, _ = correct_velocity(state, U_star, V_star, W_star, P_new)
    assert np.any(U_new == 0.0)

def test_fluid_adjacent_faces(minimal_state):
    state = minimal_state
    state["is_fluid"][1,1,1] = False
    U_star = np.ones_like(state["U"])
    P_new = np.zeros_like(state["P"])
    U_new, _, _ = correct_velocity(state, U_star, state["V"], state["W"], P_new)
    assert np.any(U_new == 0.0)

def test_minimal_grid():
    state = {
        "Mask": np.ones((1,1,1), int),
        "is_fluid": np.ones((1,1,1), bool),
        "U": np.zeros((2,1,1)),
        "V": np.zeros((1,2,1)),
        "W": np.zeros((1,1,2)),
        "P": np.zeros((1,1,1)),
        "Constants": {"rho":1,"dt":0.1},
        "Operators": {
            "gradient_p_x": lambda P,s: np.zeros((2,1,1)),
            "gradient_p_y": lambda P,s: np.zeros((1,2,1)),
            "gradient_p_z": lambda P,s: np.zeros((1,1,2)),
        }
    }
    correct_velocity(state, state["U"], state["V"], state["W"], state["P"])
