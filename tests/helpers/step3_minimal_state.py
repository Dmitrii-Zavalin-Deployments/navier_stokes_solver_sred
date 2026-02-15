# tests/helpers/step3_minimal_state.py

import numpy as np
from src.solver_state import SolverState


def make_step3_minimal_state():
    """
    Minimal valid SolverState for Step 3/4 tests.
    """
    state = SolverState()

    state.config = {}
    state.grid = {"nx": 2, "ny": 2, "nz": 2}

    state.fields = {
        "P": np.zeros((2, 2, 2)),
        "U": np.zeros((3, 2, 2)),
        "V": np.zeros((2, 3, 2)),
        "W": np.zeros((2, 2, 3)),
    }

    state.mask = np.ones((2, 2, 2), dtype=int)
    state.constants = {"rho": 1.0}
    state.boundary_conditions = {}
    state.health = {"post_correction_divergence_norm": 0.0}
    state.ppe = {}
    state.operators = {}

    return state