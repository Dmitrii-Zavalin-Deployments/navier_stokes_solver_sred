# tests/helpers/solver_step3_output_dummy.py

import numpy as np
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def make_step3_output_dummy(nx=4, ny=4, nz=4):
    """
    Canonical dummy representing the post-Step 3 SolverState.
    
    This dummy layers Step 3 data on top of the Step 2 dummy, 
    simulating the completion of the velocity prediction, 
    pressure solve, and velocity correction.
    """

    # 1. Start from the frozen Step 2 dummy (The Foundation)
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)

    # ------------------------------------------------------------------
    # 2. Update Fields (The Corrected/Projected Fields)
    # ------------------------------------------------------------------
    # In Step 3, U, V, W are updated to be divergence-free
    state.fields["U"] = np.zeros((nx + 1, ny, nz))
    state.fields["V"] = np.zeros((nx, ny + 1, nz))
    state.fields["W"] = np.zeros((nx, ny, nz + 1))
    state.fields["P"] = np.zeros((nx, ny, nz))

    # ------------------------------------------------------------------
    # 3. Add Intermediate "Star" Fields (The Predicton results)
    # ------------------------------------------------------------------
    state.intermediate_fields = {
        "U_star": np.zeros((nx + 1, ny, nz)),
        "V_star": np.zeros((nx, ny + 1, nz)),
        "W_star": np.zeros((nx, ny, nz + 1)),
    }

    # ------------------------------------------------------------------
    # 4. Update PPE Metadata (Reflecting the Solve results)
    # ------------------------------------------------------------------
    state.ppe.update({
        "iterations": 12,           # How many iterations the solver took
        "converged": True,          # Did it hit the tolerance?
        "rhs_norm": 1e-10,          # Final residual
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
    # 6. Initialize/Update History (The Time-Series Tracking)
    # ------------------------------------------------------------------
    state.history["times"].append(state.time)
    state.history["divergence_norms"].append(1e-12)
    state.history["max_velocity_history"].append(0.0)
    state.history["ppe_iterations_history"].append(12)
    state.history["energy_history"].append(0.0)

    # ------------------------------------------------------------------
    # 7. Progression Flags
    # ------------------------------------------------------------------
    state.iteration = 1
    state.time += state.constants.get("dt", 0.0)
    state.ready_for_time_loop = True 

    return state