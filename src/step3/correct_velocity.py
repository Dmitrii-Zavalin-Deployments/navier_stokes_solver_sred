# src/step3/correct_velocity.py

import numpy as np

def correct_velocity(state, U_star, V_star, W_star, P_new):
    """
    Correct velocity using the pressure gradient: u^{n+1} = u* - dt/rho * grad(p)
    Also enforces zero-velocity on faces adjacent to solid cells.
    """
    # Use config['dt'] as the source of truth for the timestep
    rho = state.constants["rho"]
    dt = state.config["dt"]

    # 1. Extract gradient operators (built in Step 2)
    grad_x = state.operators["grad_x"]
    grad_y = state.operators["grad_y"]
    grad_z = state.operators["grad_z"]

    # 2. Compute pressure gradients
    Gx = grad_x(P_new)
    Gy = grad_y(P_new)
    Gz = grad_z(P_new)

    # 3. Apply the Projection Correction
    # This subtracts the 'mass-correcting' pressure gradient
    U_new = np.array(U_star, copy=True) - (dt / rho) * Gx
    V_new = np.array(V_star, copy=True) - (dt / rho) * Gy
    W_new = np.array(W_star, copy=True) - (dt / rho) * Gz

    # 4. Zero faces adjacent to internal solid cells (The "Neighbor Rule")
    # This ensures a 'No-Penetration' condition for internal obstacles.
    is_solid = ~state.is_fluid

    # Internal Mask for U (Face sits between i and i+1)
    mask_u = is_solid[:-1, :, :] | is_solid[1:, :, :]
    U_new[mask_u] = 0.0
    
    # Internal Mask for V (Face sits between j and j+1)
    mask_v = is_solid[:, :-1, :] | is_solid[:, 1:, :]
    V_new[mask_v] = 0.0
    
    # Internal Mask for W (Face sits between k and k+1)
    mask_w = is_solid[:, :, :-1] | is_solid[:, :, 1:]
    W_new[mask_w] = 0.0

    return U_new, V_new, W_new