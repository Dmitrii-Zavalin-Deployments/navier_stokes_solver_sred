# tests/step3/test_predict_velocity.py

import numpy as np
from src.step3.predict_velocity import predict_velocity

def test_zero_ops_zero_forces(minimal_state):
    state = minimal_state
    U_star, V_star, W_star = predict_velocity(state)
    assert np.allclose(U_star, state["U"])
    assert np.allclose(V_star, state["V"])
    assert np.allclose(W_star, state["W"])

def test_constant_force(minimal_state):
    state = minimal_state
    state["Config"]["external_forces"] = {"fx": 1.0}
    U_star, V_star, W_star = predict_velocity(state)
    assert np.any(U_star != 0.0)
    assert np.allclose(V_star, 0.0)
    assert np.allclose(W_star, 0.0)

def test_solid_mask_respected(minimal_state):
    state = minimal_state
    state["Mask"][1,1,1] = 0
    state["Config"]["external_forces"] = {"fx": 1.0}
    U_star, _, _ = predict_velocity(state)
    assert np.any(U_star == 0.0)

def test_temp_buffers(minimal_state):
    state = minimal_state
    U_before = state["U"].copy()
    predict_velocity(state)
    assert np.allclose(state["U"], U_before)

def test_minimal_grid():
    state = {
        "Mask": np.ones((1,1,1), int),
        "is_fluid": np.ones((1,1,1), bool),
        "U": np.zeros((2,1,1)),
        "V": np.zeros((1,2,1)),
        "W": np.zeros((1,1,2)),
        "P": np.zeros((1,1,1)),
        "Config": {"external_forces": {}},
        "Constants": {"rho":1,"mu":0.1,"dt":0.01},
        "Operators": {
            "advection_u": lambda U,V,W,s: np.zeros_like(U),
            "advection_v": lambda U,V,W,s: np.zeros_like(V),
            "advection_w": lambda U,V,W,s: np.zeros_like(W),
            "laplacian_u": lambda U,s: np.zeros_like(U),
            "laplacian_v": lambda V,s: np.zeros_like(V),
            "laplacian_w": lambda W,s: np.zeros_like(W),
        }
    }
    predict_velocity(state)
