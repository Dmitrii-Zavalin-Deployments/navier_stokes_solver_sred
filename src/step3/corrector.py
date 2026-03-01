# src/step3/corrector.py

import numpy as np
from src.solver_state import SolverState

def correct_velocity(state: SolverState) -> None:
    """
    Step 3.3: Projection/Correction.
    Point 2: V_new = V* - (dt/rho) * grad(P)
    """
    rho = state.constants.rho
    dt = state.config.dt
    coeff = dt / rho

    # Subtract pressure gradient from intermediate velocity
    state.fields.U = state.fields.U_star - coeff * (state.operators.grad_x @ state.fields.P.ravel()).reshape(state.fields.U.shape)
    state.fields.V = state.fields.V_star - coeff * (state.operators.grad_y @ state.fields.P.ravel()).reshape(state.fields.V.shape)
    state.fields.W = state.fields.W_star - coeff * (state.operators.grad_z @ state.fields.P.ravel()).reshape(state.fields.W.shape)

    # Update Health Context (SSoT Rule 4)
    state.health.max_u = float(max(np.max(np.abs(state.fields.U)), 
                                   np.max(np.abs(state.fields.V)), 
                                   np.max(np.abs(state.fields.W))))
    
    v_new_flat = np.concatenate([state.fields.U.ravel(), state.fields.V.ravel(), state.fields.W.ravel()])
    div_new = state.operators.divergence @ v_new_flat
    state.health.divergence_norm = float(np.linalg.norm(div_new) / div_new.size)
    state.health.post_correction_divergence_norm = state.health.divergence_norm