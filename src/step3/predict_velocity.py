# src/step3/predict_velocity.py

import numpy as np

def predict_velocity(state):
    """
    Compute intermediate velocity u* using advection, diffusion, and external forces.
    Returns U_star, V_star, W_star.

    Rules enforced:
      • Apply advection, diffusion, and forces normally.
      • Zero faces adjacent to solid cells (OR logic).
      • Do NOT zero anything when all cells are fluid.
    """

    rho = state["Constants"]["rho"]
    mu = state["Constants"]["mu"]
    dt = state["Constants"]["dt"]

    U, V, W = state["U"], state["V"], state["W"]

    # Operators
    adv_u = state["Operators"]["advection_u"]
    adv_v = state["Operators"]["advection_v"]
    adv_w = state["Operators"]["advection_w"]

    lap_u = state["Operators"]["laplacian_u"]
    lap_v = state["Operators"]["laplacian_v"]
    lap_w = state["Operators"]["laplacian_w"]

    # Advection
    Au = adv_u(U, V, W, state)
    Av = adv_v(U, V, W, state)
    Aw = adv_w(U, V, W, state)

    # Diffusion
    Du = lap_u(U, state)
    Dv = lap_v(V, state)
    Dw = lap_w(W, state)

    # External forces
    forces = state["Config"].get("external_forces", {})
    fx = forces.get("fx", 0.0)
    fy = forces.get("fy", 0.0)
    fz = forces.get("fz", 0.0)

    # Compute U*, V*, W*
    U_star = U + dt * (-Au + (mu / rho) * Du + fx)
    V_star = V + dt * (-Av + (mu / rho) * Dv + fy)
    W_star = W + dt * (-Aw + (mu / rho) * Dw + fz)

    # ----------------------------------------------------------------------
    # Zero faces adjacent to solids (OR logic)
    # ----------------------------------------------------------------------

    mask = state["Mask"]
    is_solid = (mask == 0)
    nx, ny, nz = mask.shape

    # U faces: between i-1 and i
    solid_u = np.zeros_like(U_star, dtype=bool)
    solid_u[1:-1, :, :] = is_solid[:-1, :, :] | is_solid[1:, :, :]
    U_star[solid_u] = 0.0

    # V faces: between j-1 and j
    solid_v = np.zeros_like(V_star, dtype=bool)
    solid_v[:, 1:-1, :] = is_solid[:, :-1, :] | is_solid[:, 1:, :]
    V_star[solid_v] = 0.0

    # W faces: between k-1 and k
    solid_w = np.zeros_like(W_star, dtype=bool)
    solid_w[:, :, 1:-1] = is_solid[:, :, :-1] | is_solid[:, :, 1:]
    W_star[solid_w] = 0.0

    return U_star, V_star, W_star
