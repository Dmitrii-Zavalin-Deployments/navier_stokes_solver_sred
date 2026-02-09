# src/step3/predict_velocity.py

import numpy as np


def predict_velocity(state, fields):
    """
    Pure Step‑3 velocity prediction.

    Computes intermediate velocity u* using:
        • advection
        • diffusion
        • external forces

    Rules enforced:
      • Zero faces adjacent to solid cells (OR logic).
      • Do NOT zero anything when all cells are fluid.

    Inputs:
        state  – Step‑2 output dict
        fields – dict with:
                 { "U": ndarray, "V": ndarray, "W": ndarray, "P": ndarray }

    Returns:
        U_star, V_star, W_star – predicted velocities
    """

    constants = state["constants"]
    rho = constants["rho"]
    mu = constants["mu"]
    dt = constants["dt"]

    U = np.asarray(fields["U"])
    V = np.asarray(fields["V"])
    W = np.asarray(fields["W"])

    # ------------------------------------------------------------------
    # Operators (pure functions)
    # ------------------------------------------------------------------
    adv_u = state["advection"]["u"]["op"]
    adv_v = state["advection"]["v"]["op"]
    adv_w = state["advection"]["w"]["op"]

    lap_u = state["laplacians"]["u"]["op"]
    lap_v = state["laplacians"]["v"]["op"]
    lap_w = state["laplacians"]["w"]["op"]

    # ------------------------------------------------------------------
    # Advection
    # ------------------------------------------------------------------
    Au = adv_u(U, V, W)
    Av = adv_v(U, V, W)
    Aw = adv_w(U, V, W)

    # ------------------------------------------------------------------
    # Diffusion
    # ------------------------------------------------------------------
    Du = lap_u(U)
    Dv = lap_v(V)
    Dw = lap_w(W)

    # ------------------------------------------------------------------
    # External forces (optional)
    # ------------------------------------------------------------------
    forces = state["config"].get("external_forces", {})
    fx = forces.get("fx", 0.0)
    fy = forces.get("fy", 0.0)
    fz = forces.get("fz", 0.0)

    # ------------------------------------------------------------------
    # Compute U*, V*, W*
    # ------------------------------------------------------------------
    U_star = U + dt * (-Au + (mu / rho) * Du + fx)
    V_star = V + dt * (-Av + (mu / rho) * Dv + fy)
    W_star = W + dt * (-Aw + (mu / rho) * Dw + fz)

    # ------------------------------------------------------------------
    # Zero faces adjacent to solids (OR logic)
    # ------------------------------------------------------------------
    is_solid = np.asarray(state["mask_semantics"]["is_solid"], dtype=bool)

    if np.any(is_solid):
        # U faces
        solid_u = np.zeros_like(U_star, dtype=bool)
        solid_u[1:-1, :, :] = is_solid[:-1, :, :] | is_solid[1:, :, :]
        U_star = np.array(U_star, copy=True)
        U_star[solid_u] = 0.0

        # V faces
        solid_v = np.zeros_like(V_star, dtype=bool)
        solid_v[:, 1:-1, :] = is_solid[:, :-1, :] | is_solid[:, 1:, :]
        V_star = np.array(V_star, copy=True)
        V_star[solid_v] = 0.0

        # W faces
        solid_w = np.zeros_like(W_star, dtype=bool)
        solid_w[:, :, 1:-1] = is_solid[:, :, :-1] | is_solid[:, :, 1:]
        W_star = np.array(W_star, copy=True)
        W_star[solid_w] = 0.0

    return U_star, V_star, W_star
