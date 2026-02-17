# file: src/step3/predict_velocity.py

import numpy as np


def predict_velocity(state, fields):
    """
    Step‑3 velocity prediction.
    Computes intermediate velocity U* using:
        • diffusion (via Step‑2 Laplacian operators)
        • external forces (optional)
    Pure function: does not mutate state.
    """

    rho = state.constants["rho"]
    mu = state.constants["mu"]
    dt = state.constants["dt"]

    U = np.asarray(fields["U"])
    V = np.asarray(fields["V"])
    W = np.asarray(fields["W"])

    # ------------------------------------------------------------
    # 1. Diffusion operators from Step‑2
    # ------------------------------------------------------------
    lap_u = state.operators["lap_u"]
    lap_v = state.operators["lap_v"]
    lap_w = state.operators["lap_w"]

    Du = lap_u(U)
    Dv = lap_v(V)
    Dw = lap_w(W)

    # ------------------------------------------------------------
    # 2. External forces (optional)
    # ------------------------------------------------------------
    forces = state.config.get("external_forces", {})
    fx = forces.get("fx", 0.0)
    fy = forces.get("fy", 0.0)
    fz = forces.get("fz", 0.0)

    # ------------------------------------------------------------
    # 3. Compute U*, V*, W*
    # ------------------------------------------------------------
    U_star = U + dt * ((mu / rho) * Du + fx)
    V_star = V + dt * ((mu / rho) * Dv + fy)
    W_star = W + dt * ((mu / rho) * Dw + fz)

    # ------------------------------------------------------------
    # 4. Zero faces adjacent to solid cells
    # ------------------------------------------------------------
    is_fluid = state.is_fluid
    is_solid = ~is_fluid

    # U faces: between i-1 and i
    solid_u = np.zeros_like(U_star, dtype=bool)
    solid_u[1:-1, :, :] = is_solid[:-1, :, :] | is_solid[1:, :, :]
    U_star = np.array(U_star, copy=True)
    U_star[solid_u] = 0.0

    # V faces: between j-1 and j
    solid_v = np.zeros_like(V_star, dtype=bool)
    solid_v[:, 1:-1, :] = is_solid[:, :-1, :] | is_solid[:, 1:, :]
    V_star = np.array(V_star, copy=True)
    V_star[solid_v] = 0.0

    # W faces: between k-1 and k
    solid_w = np.zeros_like(W_star, dtype=bool)
    solid_w[:, :, 1:-1] = is_solid[:, :, :-1] | is_solid[:, :, 1:]
    W_star = np.array(W_star, copy=True)
    W_star[solid_w] = 0.0

    return U_star, V_star, W_star
