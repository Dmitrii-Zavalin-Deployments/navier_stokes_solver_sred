# tests/helpers/solver_step4_output_dummy.py

import numpy as np
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy

def make_step4_output_dummy(nx=4, ny=4, nz=4):
    """
    Canonical dummy representing the post-Step 4 SolverState.
    Updated for Departmental Integrity and Article 3 (Universal State Container).
    
    Step 4 adds:
      - Extended fields (P_ext, U_ext, V_ext, W_ext) with Ghost Cell padding.
      - Final boundary-enforced diagnostics.
      - Grid volume calculations derived from the 'grid' department.
    """

    # 1. Start from the Projection Snapshot (Step 3)
    # Inherits: grid["dx"], constants["dt"], iteration history, and projected fields.
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)

    # ------------------------------------------------------------------
    # 2. Add Extended "Ghost" Fields (For Visualization/BCs)
    # ------------------------------------------------------------------
    # Logic: 
    # Pressure (Centers): N + 2 ghosts (Left/Right)
    # Velocity (Faces): (N+1) faces + 2 ghosts = N + 3
    state.P_ext = np.zeros((nx + 2, ny + 2, nz + 2))
    state.U_ext = np.zeros((nx + 3, ny + 2, nz + 2))
    state.V_ext = np.zeros((nx + 2, ny + 3, nz + 2))
    state.W_ext = np.zeros((nx + 2, ny + 2, nz + 3))

    # ------------------------------------------------------------------
    # 3. Add Step 4 Specific Diagnostics (Pure Path Compliant)
    # ------------------------------------------------------------------
    # CRITICAL: We retrieve spacing from state.grid, NOT state.constants.
    dx = state.grid["dx"]
    dy = state.grid["dy"]
    dz = state.grid["dz"]

    state.step4_diagnostics = {
        "total_fluid_cells": nx * ny * nz,
        "grid_volume_per_cell": (dx * dy * dz),
        "initialized": True,
        "post_bc_max_velocity": 0.0,
        "post_bc_divergence_norm": 1e-12,
        "bc_violation_count": 0,
    }

    # ------------------------------------------------------------------
    # 4. Finalize State for Output/Loop
    # ------------------------------------------------------------------
    # Step 4 prepares the snapshot for either visualization (Step 5)
    # or cycling back to the next time step.
    state.ready_for_time_loop = True

    return state