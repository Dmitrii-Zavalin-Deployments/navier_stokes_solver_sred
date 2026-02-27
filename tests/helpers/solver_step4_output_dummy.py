# tests/helpers/solver_step4_output_dummy.py

import numpy as np
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy

def make_step4_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 4 Dummy: Mimics the state after Boundary Enforcement and Ghost Cell padding.
    
    Constitutional Role: 
    - Initializes Extended Fields (P_ext, U_ext, V_ext, W_ext).
    - Populates the Diagnostics department with pre-flight audit data.
    - Finalizes readiness for visualization or iterative cycling.
    """
    # 1. Inherit the Projection Foundation (Step 3)
    # Inherits: grid, operators, projected fields, and history.
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)

    # ------------------------------------------------------------------
    # 2. Add Extended "Ghost" Fields (FieldData Safe)
    # ------------------------------------------------------------------
    # Staggered Grid Halo Logic:
    # Pressure (Centers): nx + 2 ghosts
    # Velocity (Faces): (nx+1) faces + 2 ghosts = nx + 3
    
    state.fields.P_ext = np.zeros((nx + 2, ny + 2, nz + 2))
    state.fields.U_ext = np.zeros((nx + 3, ny + 2, nz + 2))
    state.fields.V_ext = np.zeros((nx + 2, ny + 3, nz + 2))
    state.fields.W_ext = np.zeros((nx + 2, ny + 2, nz + 3))

    # ------------------------------------------------------------------
    # 3. Populate Diagnostics (Diagnostics Safe)
    # ------------------------------------------------------------------
    # Note: We use the property setters to satisfy the ValidatedContainer.
    
    # Calculate approximate memory footprint (float64 = 8 bytes)
    total_voxels = (nx+2)*(ny+2)*(nz+2) 
    state.diagnostics.memory_footprint_gb = (total_voxels * 8 * 4) / 1e9
    
    state.diagnostics.bc_verification_passed = True
    
    # CFL Limit based on current max velocity (from Step 3 health)
    max_u = state.health.max_u
    if max_u > 0:
        state.diagnostics.initial_cfl_dt = 0.5 * state.grid.dx / max_u
    else:
        state.diagnostics.initial_cfl_dt = state.config.ppe_tolerance # Placeholder

    # ------------------------------------------------------------------
    # 4. Finalize State
    # ------------------------------------------------------------------
    # Step 4 marks the state as fully audited and ready for the next phase.
    state.ready_for_time_loop = True

    return state