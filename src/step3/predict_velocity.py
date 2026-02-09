# src/step3/predict_velocity.py

import numpy as np


def predict_velocity(state, fields):
    """
    Pure Step‑3 velocity prediction.

    Computes intermediate velocity u* using:
        • diffusion (if laplacian operators are callable)
        • external forces (optional)

    Rules enforced:
      • Zero faces adjacent to solid cells (OR logic).
      • Do NOT zero anything when all cells are fluid.

    This implementation is robust to Step‑3 dummy states where
    operators.laplacian_* are NOT callables (schema requires objects).
    In that case, diffusion is skipped.
    """

    constants = state["constants"]
    rho = constants["rho"]
    mu = constants["mu"]
    dt = constants["dt"]

    U = np.asarray(fields["U"])
    V = np.asarray(fields["V"])
    W = np.asarray(fields["W"])

    # ------------------------------------------------------------------
    # Diffusion operators (may NOT be callable in Step‑3 dummy state)
    # ------------------------------------------------------------------
    ops = state.get("operators", {})

    def get_lap_op(key):
        op = ops.get(key)
        if callable(op):
            return op
        if isinstance(op, dict) and callable(op.get("op")):
            return op["op"]

        # Fallback: zero diffusion
        return lambda arr: np.zeros_like(arr)

    lap_u = get_lap_op("laplacian_u")
    lap_v = get_lap_op("laplacian_v")
    lap_w = get_lap_op("laplacian_w")

    # Diffusion terms
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
    U_star = U + dt * ((mu / rho) * Du + fx)
    V_star = V + dt * ((mu / rho) * Dv + fy)
    W_star = W + dt * ((mu / rho) * Dw + fz)

    # ------------------------------------------------------------------
    # Zero faces adjacent to solids (OR logic)
    # ------------------------------------------------------------------
    is_solid = np.asarray(state["is_solid"], dtype=bool)

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
