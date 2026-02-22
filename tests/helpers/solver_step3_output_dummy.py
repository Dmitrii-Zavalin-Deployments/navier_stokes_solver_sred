# tests/helpers/solver_step3_output_dummy.py

import numpy as np
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def make_step3_output_dummy(nx=4, ny=4, nz=4):
    """
    Canonical dummy representing the post-Step 3 SolverState.
    Updated for Departmental Integrity and Property Robustness.
    
    Step 3 adds:
      - Intermediate fields (U, V, W) representing the 'Star' state
      - Pressure Poisson Solve results (PPE metadata updates)
      - Velocity Correction (Field updates to U, V, W)
      - Health diagnostics post-projection
    """

    # 1. Start from the Operator Foundation (Step 2)
    # Inherits: grid["total_cells"], ppe["dimension"], and Sparse Operators
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)

    # ------------------------------------------------------------------
    # 2. Update Fields (The Corrected/Projected Fields)
    # ------------------------------------------------------------------
    # These represent final divergence-free values. 
    # Staggered shapes: nx+1 for the primary direction.
    state.fields.update({
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
        "P": np.zeros((nx, ny, nz)),
    })

    # ------------------------------------------------------------------
    # 3. Add Intermediate Fields (The Predictor/Star step)
    # ------------------------------------------------------------------
    # FIX: Keys changed from 'U_star' to 'U' to match test expectations 
    # and internal department naming conventions.
    state.intermediate_fields = {
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # ------------------------------------------------------------------
    # 4. Update PPE Metadata (Reflecting Linear Algebra Solve)
    # ------------------------------------------------------------------
    state.ppe.update({
        "iterations": 12,           
        "converged": True,          
        "rhs_norm": 1e-10,          
    })

    # ------------------------------------------------------------------
    # 5. Update Health Diagnostics (Post-Correction metrics)
    # ------------------------------------------------------------------
    state.health.update({
        "post_correction_divergence_norm": 1e-12,
        "max_velocity_magnitude": 0.0,
        "cfl_advection_estimate": 0.0,
    })

    # ------------------------------------------------------------------
    # 6. Append to History (Time-series tracking)
    # ------------------------------------------------------------------
    state.history["times"].append(state.time)
    state.history["divergence_norms"].append(1e-12)
    state.history["max_velocity_history"].append(0.0)
    state.history["ppe_iterations_history"].append(12)
    state.history["energy_history"].append(0.0)

    # ------------------------------------------------------------------
    # 7. Progression Flags & Metadata
    # ------------------------------------------------------------------
    state.iteration = 1
    
    dt = state.constants.get("dt", 0.01)
    state.time += dt
    
    state.ready_for_time_loop = True 

    return state