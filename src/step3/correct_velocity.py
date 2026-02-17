# src/step3/correct_velocity.py

import numpy as np


def correct_velocity(state, U_star, V_star, W_star, P_new):
    """
    Correct velocity using the pressure gradient:

        u^{n+1} = u* - dt/rho * grad(p)

    Pure Step‑3 function: does not mutate state.
    """

    rho = state.constants["rho"]
    dt = state.constants["dt"]

    # ------------------------------------------------------------
    # 1. Extract gradient operators from Step‑2
    # ------------------------------------------------------------
    grad_x = state.operators["grad_x"]
    grad_y = state.operators["grad_y"]
    grad_z = state.operators["grad_z"]

    # ------------------------------------------------------------
    # 2. Compute pressure gradients on staggered faces
    # ------------------------------------------------------------
    Gx = grad_x(P_new)  # shape like U
    Gy = grad_y(P_new)  # shape like V
    Gz = grad_z(P_new)  # shape like W

    # ------------------------------------------------------------
    # 3. Apply correction
    # ------------------------------------------------------------
    U_new = np.array(U_star, copy=True) - (dt / rho) * Gx
    V_new = np.array(V_star, copy=True) - (dt / rho) * Gy
    W_new = np.array(W_star, copy=True) - (dt / rho) * Gz

    # ------------------------------------------------------------
    # 4. Zero faces adjacent to solid cells
    # ------------------------------------------------------------
    is_fluid = state.is_fluid
    is_solid = ~is_fluid

    # U faces: between i-1 and i
    solid_u = np.zeros_like(U_new, dtype=bool)
    solid_u[1:-1, :, :] = is_solid[:-1, :, :] | is_solid[1:, :, :]
    U_new[solid_u] = 0.0

    # V faces: between j-1 and j
    solid_v = np.zeros_like(V_new, dtype=bool)
    solid_v[:, 1:-1, :] = is_solid[:, :-1, :] | is_solid[:, 1:, :]
    V_new[solid_v] = 0.0

    # W faces: between k-1 and k
    solid_w = np.zeros_like(W_new, dtype=bool)
    solid_w[:, :, 1:-1] = is_solid[:, :, :-1] | is_solid[:, :, 1:]
    W_new[solid_w] = 0.0

    return U_new, V_new, W_new
