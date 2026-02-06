# tests/step3/test_ppe_rhs_shape.py

import numpy as np
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.predict_velocity import predict_velocity

def test_ppe_rhs_shape(minimal_state):
    state = minimal_state

    U_star, V_star, W_star = predict_velocity(state)
    rhs = build_ppe_rhs(state, U_star, V_star, W_star)

    assert rhs.shape == state["P"].shape
