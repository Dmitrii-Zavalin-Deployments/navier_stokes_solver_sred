# src/step3/correct_velocity.py

import numpy as np

def correct_velocity(state, U_star, V_star, W_star, P_new):
    """
    Correct velocity using the pressure gradient and enforce no-penetration at solid walls.
    """
    rho = state.constants["rho"]
    dt = state.config["dt"]

    # 1. Compute pressure gradients using the operators
    Gx = state.operators["grad_x"](P_new)
    Gy = state.operators["grad_y"](P_new)
    Gz = state.operators["grad_z"](P_new)

    # 2. Apply the Projection Correction
    U_new = np.array(U_star, copy=True) - (dt / rho) * Gx
    V_new = np.array(V_star, copy=True) - (dt / rho) * Gy
    W_new = np.array(W_star, copy=True) - (dt / rho) * Gz

    # 3. Enforce Zero Velocity on Solid Boundaries (Neighbor Rule)
    # We only apply this to the internal faces.
    is_solid = ~state.is_fluid
    
    # U-component: faces between (i, j, k) and (i+1, j, k)
    # Slice the velocity to match the internal mask shape (nx-1, ny, nz)
    mask_u = is_solid[:-1, :, :] | is_solid[1:, :, :]
    U_internal = U_new[1:-1, :, :]
    U_internal[mask_u] = 0.0
    U_new[1:-1, :, :] = U_internal
    
    # V-component: faces between (i, j, k) and (i, j+1, k)
    mask_v = is_solid[:, :-1, :] | is_solid[:, 1:, :]
    V_internal = V_new[:, 1:-1, :]
    V_internal[mask_v] = 0.0
    V_new[:, 1:-1, :] = V_internal
    
    # W-component: faces between (i, j, k) and (i, j, k+1)
    mask_w = is_solid[:, :, :-1] | is_solid[:, :, 1:]
    W_internal = W_new[:, :, 1:-1]
    W_internal[mask_w] = 0.0
    W_new[:, :, 1:-1] = W_internal

    return U_new, V_new, W_new