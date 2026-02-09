# src/step3/correct_velocity.py

import numpy as np


def correct_velocity(state, U_star, V_star, W_star, P_new):
    """
    Correct velocity using the pressure gradient:

        u^{n+1} = u* - dt/rho * grad(p)

    Rules enforced:
      • Apply correction everywhere EXCEPT faces adjacent to solid cells.
      • Faces adjacent to solids are zeroed (no-through condition), but only if solids exist.
      • Faces not adjacent to ANY fluid cell are also zeroed,
        but only if there exists at least one non-fluid cell.
      • No unintended zeroing occurs when all cells are fluid.

    Inputs:
        state  – Step‑2 output dict
        U_star, V_star, W_star – predicted velocities (staggered)
        P_new – corrected pressure field

    Returns:
        U_new, V_new, W_new – corrected velocities
    """

    constants = state["constants"]
    rho = constants["rho"]
    dt = constants["dt"]

    # Gradient operators (pure functions from Step‑2 structure)
    grad_px = state["pressure_gradients"]["x"]["op"]
    grad_py = state["pressure_gradients"]["y"]["op"]
    grad_pz = state["pressure_gradients"]["z"]["op"]

    # Compute pressure gradients on staggered faces
    Gx = grad_px(P_new)  # shape like U
    Gy = grad_py(P_new)  # shape like V
    Gz = grad_pz(P_new)  # shape like W

    # Apply correction (work on copies to keep everything pure)
    U_new = np.array(U_star, copy=True) - (dt / rho) * Gx
    V_new = np.array(V_star, copy=True) - (dt / rho) * Gy
    W_new = np.array(W_star, copy=True) - (dt / rho) * Gz

    mask_sem = state["mask_semantics"]
    is_solid = np.asarray(mask_sem["is_solid"], dtype=bool)
    is_fluid = np.asarray(mask_sem["is_fluid"], dtype=bool)

    # ------------------------------------------------------------------
    # 1. Zero faces adjacent to solids (OR logic), but ONLY if solids exist
    # ------------------------------------------------------------------
    if np.any(is_solid):
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

    # ------------------------------------------------------------------
    # 2. Zero faces NOT adjacent to any fluid cell,
    #    but only if there exists at least one non-fluid cell.
    #    This keeps all-fluid domains unchanged.
    # ------------------------------------------------------------------
    if np.any(~is_fluid):
        # U faces: fluid if either adjacent cell is fluid
        fluid_u = np.zeros_like(U_new, bool)
        fluid_u[1:-1, :, :] = is_fluid[:-1, :, :] | is_fluid[1:, :, :]
        U_new[~fluid_u] = 0.0

        # V faces
        fluid_v = np.zeros_like(V_new, bool)
        fluid_v[:, 1:-1, :] = is_fluid[:, :-1, :] | is_fluid[:, 1:, :]
        V_new[~fluid_v] = 0.0

        # W faces
        fluid_w = np.zeros_like(W_new, bool)
        fluid_w[:, :, 1:-1] = is_fluid[:, :, :-1] | is_fluid[:, :, 1:]
        W_new[~fluid_w] = 0.0

    return U_new, V_new, W_new
