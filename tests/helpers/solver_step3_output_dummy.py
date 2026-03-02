# tests/helpers/solver_step3_output_dummy.py

import numpy as np
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def make_step3_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 3 Dummy: Mimics the state after one full time-step iteration.
    
    Constitutional Role: 
    - Records the transition from initial state to t + dt.
    - Updates history lists (Times, Divergence, Energy).
    - Reflects the results of the Pressure Poisson solve.
    """
    # 1. Inherit the Mathematical Machine (Step 2)
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)

    # ------------------------------------------------------------------
    # 2. Update Fields (The Results of the Projection)
    # ------------------------------------------------------------------
    # After Step 3, the velocities are divergence-free (corrected)
    state.fields.U = np.zeros((nx + 1, ny, nz))
    state.fields.V = np.zeros((nx, ny + 1, nz))
    state.fields.W = np.zeros((nx, ny, nz + 1))
    state.fields.P = np.zeros((nx, ny, nz))
 
    # Predictor fields for Step 3 Projection logic
    state.fields.U_star = np.zeros((nx + 1, ny, nz))
    state.fields.V_star = np.zeros((nx, ny + 1, nz))
    state.fields.W_star = np.zeros((nx, ny, nz + 1))

    # ------------------------------------------------------------------
    # 3. Update Health (Post-Correction Vitals)
    # ------------------------------------------------------------------
    state.health.is_stable = True
    state.health.max_u = 0.5  # Non-zero speed after movement
    state.health.divergence_norm = 1e-14 # Mass conservation proof
    state.health.post_correction_divergence_norm = 1e-14

    # ------------------------------------------------------------------
    # 4. Append to History (The Black Box Recorder)
    # ------------------------------------------------------------------
    # Note: History lists are initialized in SolverState.
    # We append the first 'snapshot' of data.
    state.history.times.append(state.time)
    state.history.divergence_norms.append(1e-14)
    state.history.max_velocity_history.append(0.5)
    state.history.energy_history.append(0.0125)
    # ppe_status_history tracks solve outcomes (e.g., 'converged')
    state.history.ppe_status_history.append("converged")

    # ------------------------------------------------------------------
    # 5. Progression Flags
    # ------------------------------------------------------------------
    state.iteration = 1
    # dt is usually set in config/grid properties; we increment physical time
    state.time += state.config.dt
        
    # After Step 3, we have successfully run a loop, so this is definitely True
    state.ready_for_time_loop = True 

    return state