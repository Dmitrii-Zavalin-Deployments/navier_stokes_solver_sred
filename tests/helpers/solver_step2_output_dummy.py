# tests/helpers/solver_step2_output_dummy.py

"""
Static, canonical Step 2 dummy output.

This file defines a fully realistic SolverState object that matches the
expected structure after Step 2 orchestration has completed in production.

It mirrors the Step 1 dummy, but with all Step 2 additions:
- mask semantics (is_fluid, is_boundary_cell)
- constants (inv_dx, inv_dy, inv_dz, inv_dx2, inv_dy2, inv_dz2)
- operators (divergence, gradients, laplacians, advection)
- PPE structure (rhs_builder, ppe_is_singular)
- health diagnostics (divergence_norm, max_velocity, cfl)
"""

import numpy as np
from types import SimpleNamespace


def solver_step2_output_dummy():
    nx, ny, nz = 4, 4, 4
    dx = 1.0
    dt = 0.1
    rho = 1.0

    # ------------------------------------------------------------
    # Base structure (same as Step 1 dummy)
    # ------------------------------------------------------------
    state = SimpleNamespace()

    state.grid = SimpleNamespace(
        nx=nx, ny=ny, nz=nz,
        dx=dx, dy=dx, dz=dx
    )

    state.config = SimpleNamespace(
        dt=dt,
        advection_scheme="upwind"
    )

    state.constants = {
        "rho": rho,
        "inv_dx": 1.0 / dx,
        "inv_dy": 1.0 / dx,
        "inv_dz": 1.0 / dx,
        "inv_dx2": 1.0 / (dx * dx),
        "inv_dy2": 1.0 / (dx * dx),
        "inv_dz2": 1.0 / (dx * dx),
    }

    # ------------------------------------------------------------
    # Mask semantics
    # ------------------------------------------------------------
    state.mask = np.ones((nx, ny, nz), dtype=int)
    state.is_fluid = np.ones((nx, ny, nz), dtype=bool)
    state.is_boundary_cell = np.zeros((nx, ny, nz), dtype=bool)

    # ------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # ------------------------------------------------------------
    # Operators (placeholders)
    # ------------------------------------------------------------
    state.operators = {
        "divergence": lambda fields: np.zeros((nx, ny, nz)),
        "grad_x": lambda P: np.zeros((nx + 1, ny, nz)),
        "grad_y": lambda P: np.zeros((nx, ny + 1, nz)),
        "grad_z": lambda P: np.zeros((nx, ny, nz + 1)),
        "lap_u": lambda U: np.zeros((nx + 1, ny, nz)),
        "lap_v": lambda V: np.zeros((nx, ny + 1, nz)),
        "lap_w": lambda W: np.zeros((nx, ny, nz + 1)),
        "adv_u": lambda fields: np.zeros((nx + 1, ny, nz)),
        "adv_v": lambda fields: np.zeros((nx, ny + 1, nz)),
        "adv_w": lambda fields: np.zeros((nx, ny, nz + 1)),
    }

    # ------------------------------------------------------------
    # PPE structure
    # ------------------------------------------------------------
    state.ppe = {
        "rhs_builder": lambda divergence: -rho / dt * divergence,
        "ppe_is_singular": True,
    }

    # ------------------------------------------------------------
    # Health diagnostics
    # ------------------------------------------------------------
    state.health = {
        "divergence_norm": 0.0,
        "max_velocity": 0.0,
        "cfl": 0.0,
    }

    # ------------------------------------------------------------
    # Boundary conditions (empty but valid)
    # ------------------------------------------------------------
    state.boundary_conditions = {}

    return state
