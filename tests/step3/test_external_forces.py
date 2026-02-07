# tests/step3/test_external_forces.py

import numpy as np
from src.step3.predict_velocity import predict_velocity

def test_external_forces_modify_velocity(minimal_state):
    state = minimal_state

    # Apply a simple force in x-direction
    state["Config"]["external_forces"] = {"fx": 1.0, "fy": 0.0, "fz": 0.0}

    U0 = state["U"].copy()
    U_star, V_star, W_star = predict_velocity(state)

    # U* must increase due to fx
    assert np.any(U_star > U0)
