# tests/step3/test_solve_pressure_singular.py

import numpy as np
from src.step3.solve_pressure import solve_pressure

def test_solve_pressure_singular(minimal_state):
    state = minimal_state
    state["PPE"]["ppe_is_singular"] = True

    # Create a pressure field with non-zero mean
    rhs = np.ones_like(state["P"])
    P_new = solve_pressure(state, rhs)

    fluid = state["is_fluid"]

    # Mean must be zero after singular solve
    assert abs(P_new[fluid].mean()) < 1e-12
