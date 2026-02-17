# tests/helpers/solver_step3_output_dummy.py

import numpy as np
from src.solver_state import SolverState


def make_step3_output_dummy():
    """
    Canonical dummy representing REAL production output of Step 3.
    This includes Step 1 + Step 2 + Step 3 fields exactly as they appear
    after orchestrate_step3_state() completes.
    """

    nx, ny, nz = 4, 4, 4

    # ------------------------------------------------------------------
    # Step 1 fields
    # ------------------------------------------------------------------
    config = {
        "external_forces": {"fx": 0.0, "fy": 0.0, "fz": 0.0},
    }

    grid = {"nx": nx, "ny": ny, "nz": nz}

    constants = {
        "rho": 1.0,
        "mu": 1.0,
        "dt": 0.1,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
    }

    # Staggered velocity + pressure
    fields = {
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
        "P": np.zeros((nx, ny, nz)),
    }

    mask = np.ones((nx, ny, nz), dtype=int)
    is_fluid = mask == 1
    is_boundary_cell = np.zeros_like(mask, dtype=bool)

    # ------------------------------------------------------------------
    # Step 2 fields
    # ------------------------------------------------------------------
    operators = {
        "lap_u": lambda arr: np.zeros_like(arr),
        "lap_v": lambda arr: np.zeros_like(arr),
        "lap_w": lambda arr: np.zeros_like(arr),
        "grad_x": lambda p: np.zeros((nx + 1, ny, nz)),
        "grad_y": lambda p: np.zeros((nx, ny + 1, nz)),
        "grad_z": lambda p: np.zeros((nx, ny, nz + 1)),
        "divergence": lambda U, V, W: np.zeros((nx, ny, nz)),
    }

    ppe = {
        "solver_type": "SOR",
        "tolerance": 1e-6,
        "max_iterations": 50,
        "ppe_is_singular": True,
        "rhs_builder": "rho/dt * div(U*)",
    }

    # ------------------------------------------------------------------
    # Step 3 fields (post-projection)
    # ------------------------------------------------------------------
    # After Step 3, fields are updated. Dummy uses zeros for simplicity.
    health = {
        "post_correction_divergence_norm": 0.0,
        "max_velocity_magnitude": 0.0,
        "cfl_advection_estimate": 0.0,
    }

    history = {
        "times": [0.0],
        "divergence_norms": [0.0],
        "max_velocity_history": [0.0],
        "ppe_iterations_history": [0],
        "energy_history": [0.0],
    }

    return SolverState(
        config=config,
        grid=grid,
        fields=fields,
        mask=mask,
        is_fluid=is_fluid,
        is_boundary_cell=is_boundary_cell,
        constants=constants,
        boundary_conditions=None,
        operators=operators,
        ppe=ppe,
        health=health,
        history=history,
    )
