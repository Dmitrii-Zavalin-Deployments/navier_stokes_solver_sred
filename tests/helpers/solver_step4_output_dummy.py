import numpy as np
from src.solver_state import SolverState


def make_step4_output_dummy(nx=4, ny=4, nz=4):
    """
    Production-accurate Step 4 dummy.
    Produces a fully valid SolverState exactly as it appears after Step 4.
    """

    state = SolverState()

    # ---------------------------------------------------------
    # 1. config
    # ---------------------------------------------------------
    state.config = {
        "domain": {"nx": nx, "ny": ny, "nz": nz},
        "boundary_conditions": [],
    }

    # ---------------------------------------------------------
    # 2. grid (production shape)
    # ---------------------------------------------------------
    state.grid = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
    }

    # ---------------------------------------------------------
    # 3. interior fields (Step 3 output)
    # ---------------------------------------------------------
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # ---------------------------------------------------------
    # 4. mask semantics (modern)
    # ---------------------------------------------------------
    state.mask = np.ones((nx, ny, nz), dtype=bool)
    state.is_fluid = np.ones((nx, ny, nz), dtype=bool)
    state.is_boundary_cell = np.zeros((nx, ny, nz), dtype=bool)

    # ---------------------------------------------------------
    # 5. constants
    # ---------------------------------------------------------
    state.constants = {
        "rho": 1.0,
        "mu": 0.01,
        "dt": 0.1,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
    }

    # ---------------------------------------------------------
    # 6. operators (optional in final schema)
    # ---------------------------------------------------------
    state.operators = {
        "adv_u": lambda U, V, W: np.zeros_like(U),
        "adv_v": lambda U, V, W: np.zeros_like(V),
        "adv_w": lambda U, V, W: np.zeros_like(W),
        "lap_u": lambda U: np.zeros_like(U),
        "lap_v": lambda V: np.zeros_like(V),
        "lap_w": lambda W: np.zeros_like(W),
        "divergence": lambda U, V, W: np.zeros((nx, ny, nz)),
        "grad_x": lambda P: np.zeros((nx + 1, ny, nz)),
        "grad_y": lambda P: np.zeros((nx, ny + 1, nz)),
        "grad_z": lambda P: np.zeros((nx, ny, nz + 1)),
    }

    # ---------------------------------------------------------
    # 7. PPE metadata (optional in final schema)
    # ---------------------------------------------------------
    state.ppe = {
        "solver": None,
        "tolerance": 1e-6,
        "max_iterations": 100,
        "ppe_is_singular": False,
    }

    # ---------------------------------------------------------
    # 8. Step 3 health (optional in final schema)
    # ---------------------------------------------------------
    state.health = {
        "post_correction_divergence_norm": 0.0,
        "max_velocity_magnitude": 0.0,
        "cfl_advection_estimate": 0.0,
    }

    # ---------------------------------------------------------
    # 9. Step 3 history (NOT part of final schema — keep internal only)
    # ---------------------------------------------------------
    state.history = {
        "times": [],
        "divergence_norms": [],
        "max_velocity_history": [],
        "ppe_iterations_history": [],
        "energy_history": [],
    }

    # ---------------------------------------------------------
    # 10. Step 4 extended fields
    # ---------------------------------------------------------
    state.P_ext = np.zeros((nx + 2, ny + 2, nz + 2))
    state.U_ext = np.zeros((nx + 3, ny + 2, nz + 2))
    state.V_ext = np.zeros((nx + 2, ny + 3, nz + 2))
    state.W_ext = np.zeros((nx + 2, ny + 2, nz + 3))

    # ---------------------------------------------------------
    # 11. Step 4 diagnostics
    # ---------------------------------------------------------
    state.step4_diagnostics = {
        "total_fluid_cells": nx * ny * nz,
        "grid_volume_per_cell": 1.0,
        "initialized": True,
        "post_bc_max_velocity": 0.0,
        "post_bc_divergence_norm": 0.0,
        "bc_violation_count": 0,
    }

    # ---------------------------------------------------------
    # 12. Ready flag (Step 4 sets False)
    # ---------------------------------------------------------
    state.ready_for_time_loop = False

    return state
