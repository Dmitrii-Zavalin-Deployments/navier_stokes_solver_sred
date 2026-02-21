# src/step3/correct_velocity.py

import numpy as np

def correct_velocity(state, U_star, V_star, W_star, P_new):
    """
    Correct velocity using the pressure gradient and enforce no-penetration at solid walls.
    Equation: u^{n+1} = u* - (dt/rho) * grad(p)
    """
    # 1. Fetch Physical Constants (Strict access to prevent silent errors)
    rho = state.constants["rho"]
    dt = state.config["dt"]

    # 2. Compute pressure gradients using the pre-built Step 2 operators
    Gx = state.operators["grad_x"](P_new)
    Gy = state.operators["grad_y"](P_new)
    Gz = state.operators["grad_z"](P_new)

    # 3. Apply the Projection Correction
    U_new = np.array(U_star, copy=True) - (dt / rho) * Gx
    V_new = np.array(V_star, copy=True) - (dt / rho) * Gy
    W_new = np.array(W_star, copy=True) - (dt / rho) * Gz

    # 4. Enforce Zero Velocity on Internal Solid Boundaries (Neighbor Rule)
    # A face velocity is zeroed if either cell it separates is solid.
    is_solid = ~state.is_fluid
    
    # --- U-component: internal faces in X ---
    # Sliced to (nx-1, ny, nz)
    mask_u = is_solid[:-1, :, :] | is_solid[1:, :, :]
    U_new[1:-1, :, :][mask_u] = 0.0
    
    # --- V-component: internal faces in Y ---
    # Sliced to (nx, ny-1, nz)
    mask_v = is_solid[:, :-1, :] | is_solid[:, 1:, :]
    V_new[:, 1:-1, :][mask_v] = 0.0
    
    # --- W-component: internal faces in Z ---
    # Sliced to (nx, ny, nz-1)
    mask_w = is_solid[:, :, :-1] | is_solid[:, :, 1:]
    W_new[:, :, 1:-1][mask_w] = 0.0

    return U_new, V_new, W_new