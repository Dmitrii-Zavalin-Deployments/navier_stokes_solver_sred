# src/step3/build_ppe_rhs.py

import numpy as np

def build_ppe_rhs(state, U_star, V_star, W_star):
    """Computes: rhs = (rho/dt) * divergence(U_star)."""
    rho, dt = state.constants["rho"], state.constants["dt"]
    div_op = state.operators["divergence"]

    # Flatten and concatenate staggered fields
    velocity_vector = np.concatenate([U_star.ravel(), V_star.ravel(), W_star.ravel()])

    # Matrix multiplication for divergence
    div_flat = div_op @ velocity_vector
    div = div_flat.reshape(state.fields["P"].shape)

    rhs = (rho / dt) * div
    if state.is_fluid is not None:
        rhs = np.where(state.is_fluid, rhs, 0.0)

    return rhs