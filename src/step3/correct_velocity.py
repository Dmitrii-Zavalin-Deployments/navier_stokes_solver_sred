# src/step3/correct_velocity.py

import numpy as np


def correct_velocity(state, U_star, V_star, W_star, P_new):
    """
    Correct velocity using the pressure gradient:

        u^{n+1} = u* - dt/rho * grad(p)

    Contract-test rules:
      • If pressure_gradients are missing (dummy Step‑3 state),
        apply ZERO correction.
      • Faces adjacent to solids are zeroed (no-through condition), but only if solids exist.
      • Faces not adjacent to ANY fluid cell are also zeroed,
        but only if there exists at least one non-fluid cell.
      • No unintended zeroing occurs when all cells are fluid.
    """

    constants = state["constants"]
    rho = constants["rho"]
    dt = constants["dt"]

    # ------------------------------------------------------------
    # 1. Handle missing pressure_gradients (dummy Step‑3 state)
    # ------------------------------------------------------------
    if "pressure_gradients" not in state:
        # No gradient operators → zero correction
        return (
            np.array(U_star, copy=True),
            np.array(V_star, copy=True),
            np.array(W_star, copy=True),
        )

    # ------------------------------------------------------------
    # 2. Extract gradient operators (real Step‑2 output)
    # ------------------------------------------------------------
    pg = state["pressure_gradients"]

    def _extract_op(entry):
        if isinstance(entry, dict) and callable(entry.get("op")):
            return entry["op"]
        if callable(entry):
            return entry
        # Fallback: zero operator
        return lambda arr: np.zeros_like(arr)

    grad_px = _extract_op(pg.get("x"))
    grad_py = _extract_op(pg.get("y"))
    grad_pz = _extract_op(pg.get("z"))

    # ------------------------------------------------------------
    # 3. Compute pressure gradients on staggered faces
    # ------------------------------------------------------------
    Gx = grad_px(P_new)  # shape like U
    Gy = grad_py(P_new)  # shape like V
    Gz = grad_pz(P_new)  # shape like W

    # ------------------------------------------------------------
    # 4. Apply correction
    # ------------------------------------------------------------
    U_new = np.array(U_star, copy=True) - (dt / rho) * Gx
    V_new = np.array(V_star, copy=True) - (dt / rho) * Gy
    W_new = np.array(W_star, copy=True) - (dt / rho) * Gz

    # ------------------------------------------------------------
    # 5. Solid/fluid masks
    # ------------------------------------------------------------
    mask_sem = state.get("mask_semantics", {})
    is_solid = np.asarray(mask_sem.get("is_solid", np.zeros_like(P_new)), dtype=bool)
    is_fluid = np.asarray(mask_sem.get("is_fluid", np.ones_like(P_new)), dtype=bool)

    # ------------------------------------------------------------------
    # 6. Zero faces adjacent to solids (OR logic), but ONLY if solids exist
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
    # 7. Zero faces NOT adjacent to any fluid cell,
    #    but only if there exists at least one non-fluid cell.
    # ------------------------------------------------------------------
    if np.any(~is_fluid):
        # U faces
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
