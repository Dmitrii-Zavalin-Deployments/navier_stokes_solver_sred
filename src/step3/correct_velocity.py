# src/step3/correct_velocity.py

import numpy as np

def correct_velocity(state, U_star, V_star, W_star, P_new):
    """
    Correct velocity using the pressure gradient:
        u^{n+1} = u* - dt/rho * grad(p)

    Logic:
    1. Computes pressure gradients using Step-2 operators.
    2. Subtracts the gradient from the intermediate (star) velocity.
    3. Enforces the Internal Solid Mask (neighbor rule) to prevent leakage.
    """

    rho = state.constants["rho"]
    dt = state.constants["dt"]

    # 1. Extract gradient operators from Step‑2
    grad_x = state.operators["grad_x"]
    grad_y = state.operators["grad_y"]
    grad_z = state.operators["grad_z"]

    # 2. Compute pressure gradients on staggered faces
    Gx = grad_x(P_new)  # matches shape of U
    Gy = grad_y(P_new)  # matches shape of V
    Gz = grad_z(P_new)  # matches shape of W

    # 3. Apply the Projection Correction
    # P_new already accounts for boundary conditions via the PPE solve
    U_new = np.array(U_star, copy=True) - (dt / rho) * Gx
    V_new = np.array(V_star, copy=True) - (dt / rho) * Gy
    W_new = np.array(W_star, copy=True) - (dt / rho) * Gz

    # 4. Zero faces adjacent to internal solid cells (The "Neighbor Rule")
    # This prevents fluid from entering/leaving solid obstacles
    is_solid = ~state.is_fluid

    # Internal Mask for U
    U_new[1:-1, :, :][is_solid[:-1, :, :] | is_solid[1:, :, :]] = 0.0
    
    # Internal Mask for V
    V_new[:, 1:-1, :][is_solid[:, :-1, :] | is_solid[:, 1:, :]] = 0.0
    
    # Internal Mask for W
    W_new[:, :, 1:-1][is_solid[:, :, :-1] | is_solid[:, :, 1:]] = 0.0

    return U_new, V_new, W_new