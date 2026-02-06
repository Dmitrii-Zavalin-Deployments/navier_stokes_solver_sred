# tests/step3/test_solve_pressure_non_singular.py

import numpy as np
from src.step3.solve_pressure import solve_pressure

def test_solve_pressure_non_singular(minimal_state):
    state = minimal_state
    state["PPE"]["ppe_is_singular"] = False

    rhs = np.ones_like(state["P"])
    P_new = solve_pressure(state, rhs)

    # Dummy solver returns zero pressure for non-singular case
    assert np.allclose(P_new, 0.0)
