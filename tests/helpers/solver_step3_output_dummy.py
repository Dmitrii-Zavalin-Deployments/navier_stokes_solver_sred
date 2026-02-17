# tests/helpers/solver_step3_output_dummy.py

import numpy as np
from src.solver_state import SolverState
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def make_step3_output_dummy(nx=4, ny=4, nz=4):
    """
    Canonical dummy representing the EXACT structure of a real post‑Step‑3 SolverState.

    Step 3 adds:
      - updated fields (U, V, W, P)
      - updated PPE metadata (iterations, convergence flag)
      - updated health diagnostics
      - updated history (one appended entry)

    Step 3 does NOT add:
      - new top-level attributes
      - new structural fields
    """

    # Start from Step 2 dummy
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)

    # ------------------------------------------------------------------
    # Step 3: Updated fields (projection applied)
    # ------------------------------------------------------------------
    state.fields = {
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
        "P": np.zeros((nx, ny, nz)),
    }

    # ------------------------------------------------------------------
    # Step 3: PPE metadata (updated after solve)
    # ------------------------------------------------------------------
    state.ppe.update({
        "iterations": 12,
        "converged": True,
        "ppe_is_singular": True,  # enclosed box case
    })

    # ------------------------------------------------------------------
    # Step 3: Health diagnostics (post‑correction)
    # ------------------------------------------------------------------
    state.health = {
        "post_correction_divergence_norm": 0.0,
        "max_velocity_magnitude": 0.0,
        "cfl_advection_estimate": 0.0,
    }

    # ------------------------------------------------------------------
    # Step 3: History (one appended entry)
    # ------------------------------------------------------------------
    state.history = {
        "times": [0.0],
        "divergence_norms": [0.0],
        "max_velocity_history": [0.0],
        "ppe_iterations_history": [12],
        "energy_history": [0.0],
    }

    return state
