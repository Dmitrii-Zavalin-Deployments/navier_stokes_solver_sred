# tests/helpers/solver_step3_output_dummy.py

import numpy as np
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def make_step3_output_dummy(nx=4, ny=4, nz=4):
    """
    Canonical dummy representing the post-Step 3 SolverState.
    Updated for Departmental Integrity, Staggered Grid, and Density Activation.
    
    Step 3 adds:
      - Density activation in fluid_properties (Required for projection scaling)
      - Intermediate fields (U_star, V_star, W_star) representing the 'Star' state
      - Step 3 Diagnostics: Confirms external force application (The Momentum receipt)
      - Pressure Poisson Solve results (PPE metadata updates)
      - Velocity Correction (Field updates to U, V, W)
      - Health diagnostics post-projection
    """

    # 1. Start from the Operator Foundation (Step 2)
    # Inherits: grid, external_forces, ppe["dimension"], and JSON-safe masks
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)

    # ------------------------------------------------------------------
    # 2. Activate Fluid Physics (Required for Step 3 Projection Scaling)
    # ------------------------------------------------------------------
    state.fluid_properties.update({
        "density": 1000.0,
        "viscosity": 1e-3,
    })

    # ------------------------------------------------------------------
    # 3. Update Fields (The Corrected/Projected Fields)
    # ------------------------------------------------------------------
    state.fields.update({
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
        "P": np.zeros((nx, ny, nz)),
    })

    # ------------------------------------------------------------------
    # 4. Add Intermediate Fields (The Predictor/Star step)
    # ------------------------------------------------------------------
    # Using '_star' naming convention to differentiate from final corrected fields
    state.intermediate_fields = {
        "U_star": np.zeros((nx + 1, ny, nz)),
        "V_star": np.zeros((nx, ny + 1, nz)),
        "W_star": np.zeros((nx, ny, nz + 1)),
    }

    # ------------------------------------------------------------------
    # 5. NEW: Step 3 Diagnostics (The "Execution Proof")
    # ------------------------------------------------------------------
    # This specifically fixes the test_external_forces_integrity.py failure
    state.step3_diagnostics = {
        "source_term_applied": True,           
        "force_magnitude_applied": state.constants.get("g", 9.81),
        "convection_scheme": "upwind",
        "diffusion_scheme": "central_difference"
    }

    # ------------------------------------------------------------------
    # 6. Update PPE Metadata (Reflecting Linear Algebra Solve)
    # ------------------------------------------------------------------
    state.ppe.update({
        "iterations": 12,           
        "converged": True,          
        "rhs_norm": 1e-10,          
    })

    # ------------------------------------------------------------------
    # 7. Update Health Diagnostics (Post-Correction metrics)
    # ------------------------------------------------------------------
    state.health.update({
        "post_correction_divergence_norm": 1e-12,
        "max_velocity_magnitude": 0.0,
        "cfl_advection_estimate": 0.0,
    })

    # ------------------------------------------------------------------
    # 8. Append to History (Time-series tracking)
    # ------------------------------------------------------------------
    state.history["times"].append(state.time)
    state.history["divergence_norms"].append(1e-12)
    state.history["max_velocity_history"].append(0.0)
    state.history["ppe_iterations_history"].append(12)
    state.history["energy_history"].append(0.0)

    # ------------------------------------------------------------------
    # 9. Progression Flags & Metadata
    # ------------------------------------------------------------------
    state.iteration = 1
    
    # Calculate next time step safely
    dt = state.constants.get("dt", 0.01)
    state.time += dt
    
    state.ready_for_time_loop = True 

    return state