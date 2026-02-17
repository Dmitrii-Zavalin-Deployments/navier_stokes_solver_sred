# tests/helpers/solver_step2_output_dummy.py

import numpy as np
from src.solver_state import SolverState
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Canonical dummy representing the EXACT structure of a real post‑Step‑2 SolverState.

    Step 2 adds:
      - is_fluid
      - is_boundary_cell
      - operators (div, grad, laplacians)
      - PPE metadata (solver type, tolerance, max_iterations)
      - Step‑2 health diagnostics (divergence_norm, max_velocity, cfl)

    Step 2 does NOT add:
      - pressure solve
      - velocity correction
      - history entries
    """

    # Start from Step 1 dummy
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)

    # ------------------------------------------------------------------
    # Mask semantics (Step 2 defines these explicitly)
    # ------------------------------------------------------------------
    state.is_fluid = state.mask == 1
    state.is_boundary_cell = np.zeros_like(state.mask, dtype=bool)

    # ------------------------------------------------------------------
    # Operators (placeholders but structurally correct)
    # ------------------------------------------------------------------
    state.operators = {
        "divergence": lambda U, V, W: np.zeros((nx, ny, nz)),
        "grad_x": lambda P: np.zeros((nx + 1, ny, nz)),
        "grad_y": lambda P: np.zeros((nx, ny + 1, nz)),
        "grad_z": lambda P: np.zeros((nx, ny, nz + 1)),
        "lap_u": lambda U: np.zeros((nx + 1, ny, nz)),
        "lap_v": lambda V: np.zeros((nx, ny + 1, nz)),
        "lap_w": lambda W: np.zeros((nx, ny, nz + 1)),
    }

    # ------------------------------------------------------------------
    # PPE metadata (Step 2 introduces this)
    # ------------------------------------------------------------------
    state.ppe = {
        "solver_type": "SOR",
        "tolerance": 1e-6,
        "max_iterations": 50,
        "ppe_is_singular": True,
        "rhs_builder": "rho/dt * div(U*)",
    }

    # ------------------------------------------------------------------
    # Step‑2 health diagnostics
    # ------------------------------------------------------------------
    state.health = {
        "divergence_norm": 0.0,
        "max_velocity": 0.0,
        "cfl": 0.0,
    }

    # ------------------------------------------------------------------
    # History remains empty at Step 2
    # ------------------------------------------------------------------
    state.history = {}

    return state
