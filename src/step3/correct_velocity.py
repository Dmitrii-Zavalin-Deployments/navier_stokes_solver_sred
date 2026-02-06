# src/step3/correct_velocity.py

import numpy as np

def correct_velocity(state, U_star, V_star, W_star, P_new):
    """
    Correct velocity using pressure gradient.
    Only correct faces adjacent to at least one fluid cell.
    """

    rho = state["Constants"]["rho"]
    dt = state["Constants"]["dt"]

    grad_px = state["Operators"]["gradient_p_x"]
    grad_py = state["Operators"]["gradient_p_y"]
    grad_pz = state["Operators"]["gradient_p_z"]

    Gx = grad_px(P_new, state)
    Gy = grad_py(P_new, state)
    Gz = grad_pz(P_new, state)

    U_new = U_star - (dt / rho) * Gx
    V_new = V_star - (dt / rho) * Gy
    W_new = W_star - (dt / rho) * Gz

    mask = state["Mask"]
    is_fluid = state["is_fluid"]
    is_solid = (mask == 0)

    nx, ny, nz = mask.shape

    fluid_u = np.zeros_like(U_new, bool)
    fluid_u[1:-1] = is_fluid[:-1] | is_fluid[1:]
    U_new[~fluid_u] = 0.0

    fluid_v = np.zeros_like(V_new, bool)
    fluid_v[:, 1:-1] = is_fluid[:, :-1] | is_fluid[:, 1:]
    V_new[~fluid_v] = 0.0

    fluid_w = np.zeros_like(W_new, bool)
    fluid_w[:, :, 1:-1] = is_fluid[:, :, :-1] | is_fluid[:, :, 1:]
    W_new[~fluid_w] = 0.0

    return U_new, V_new, W_new
