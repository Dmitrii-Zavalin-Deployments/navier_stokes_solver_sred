# tests/helpers/solver_step4_output_dummy.py

import numpy as np
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy

def make_step4_output_dummy(nx=4, ny=4, nz=4):
    """
    Canonical dummy representing the post-Step 4 SolverState.
    
    This dummy layers Step 4 results on top of the Step 3 foundation.
    It simulates a state where boundary conditions have been applied to 
    extended fields and diagnostics have been compiled.
    """

    # 1. Start from the post-Step 3 dummy (The Foundation)
    # This provides the corrected fields (U, V, W, P), constants, 
    # PPE solve results, and the iteration history.
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)

    # ------------------------------------------------------------------
    # 2. Add Extended "Ghost" Fields (For Visualization/BCs)
    # ------------------------------------------------------------------
    # Step 4 creates padded versions of the fields to handle 
    # boundary/ghost cell logic for external file exports.
    state.P_ext = np.zeros((nx + 2, ny + 2, nz + 2))
    state.U_ext = np.zeros((nx + 3, ny + 2, nz + 2))
    state.V_ext = np.zeros((nx + 2, ny + 3, nz + 2))
    state.W_ext = np.zeros((nx + 2, ny + 2, nz + 3))

    # ------------------------------------------------------------------
    # 3. Add Step 4 Specific Diagnostics
    # ------------------------------------------------------------------
    # These metrics are calculated specifically during Step 4's 
    # boundary condition enforcement and health checks.
    state.step4_diagnostics = {
        "total_fluid_cells": nx * ny * nz,
        "grid_volume_per_cell": (state.constants["dx"] * state.constants["dy"] * state.constants["dz"]),
        "initialized": True,
        "post_bc_max_velocity": 0.0,
        "post_bc_divergence_norm": 1e-12,
        "bc_violation_count": 0,
    }

    # ------------------------------------------------------------------
    # 4. Finalize State for Output/Loop
    # ------------------------------------------------------------------
    # Step 4 prepares the snapshot. We set the flag to False temporarily 
    # if the orchestrator needs to wait for Step 5 logging, or True 
    # if we are ready to cycle back to Step 2.
    state.ready_for_time_loop = True

    return state