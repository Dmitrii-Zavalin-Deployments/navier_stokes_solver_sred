# src/step3/correct_velocity.py

import numpy as np

def correct_velocity(state, U_star, V_star, W_star, P_new):
    """
    Correct velocity using the pressure gradient: u^{n+1} = u* - dt/rho * grad(p)
    """

    rho = state.constants["rho"]
    dt = state.constants["dt"]

    # 1. Extract gradient operators
    grad_x = state.operators["grad_x"]
    grad_y = state.operators["grad_y"]
    grad_z = state.operators["grad_z"]

    # 2. Compute pressure gradients
    Gx = grad_x(P_new)
    Gy = grad_y(P_new)
    Gz = grad_z(P_new)

    # 3. Apply the Projection Correction
    U_new = np.array(U_star, copy=True) - (dt / rho) * Gx
    V_new = np.array(V_star, copy=True) - (dt / rho) * Gy
    W_new = np.array(W_star, copy=True) - (dt / rho) * Gz

    # 4. Zero faces adjacent to internal solid cells (The "Neighbor Rule")
    is_solid = ~state.is_fluid

    # Internal Mask for U
    mask_u = is_solid[:-1, :, :] | is_solid[1:, :, :]
    U_new[1:-1, :, :][mask_u] = 0.0
    
    # Internal Mask for V
    mask_v = is_solid[:, :-1, :] | is_solid[:, 1:, :]
    V_new[:, 1:-1, :][mask_v] = 0.0
    
    # Internal Mask for W
    mask_w = is_solid[:, :, :-1] | is_solid[:, :, 1:]
    W_new[:, :, 1:-1][mask_w] = 0.0

    return U_new, V_new, W_new