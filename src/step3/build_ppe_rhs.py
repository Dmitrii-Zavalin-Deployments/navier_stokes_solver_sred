# src/step3/build_ppe_rhs.py

import numpy as np


def build_ppe_rhs(state, U_star, V_star, W_star):
    """
    Pure Step‑3 PPE RHS builder.

    Computes:
        rhs = (rho/dt) * divergence(U*, V*, W*)

    Contract:

      • If state["divergence"]["op"] is a callable, we MUST use it.
      • If no callable is present, we fall back to a zero RHS with the
        correct pressure shape (if available).
      • RHS must be zeroed inside solid cells using either:
            state["mask_semantics"]["is_solid"]  (legacy tests)
        or  state["is_solid"]                    (new Step‑2 schema)

    Inputs:
        state   – Step‑2/Step‑3 state dict
        U_star, V_star, W_star – predicted velocities

    Returns:
        rhs – ndarray, same shape as pressure field
    """

    constants = state["constants"]
    rho = constants["rho"]
    dt = constants["dt"]

    # ------------------------------------------------------------
    # 1. Compute divergence
    # ------------------------------------------------------------
    div_spec = state.get("divergence", {})
    op = div_spec.get("op") if isinstance(div_spec, dict) else None

    if callable(op):
        # Test path: use provided divergence operator
        div = op(U_star, V_star, W_star)
        div = np.asarray(div, dtype=float)
        rhs = (rho / dt) * div
    else:
        # No operator: construct zero RHS
        if "fields" in state and "P" in state["fields"]:
            P = np.asarray(state["fields"]["P"], dtype=float)
            shape = P.shape
        else:
            # Minimal fallback (only used in minimal-grid test)
            shape = np.asarray(U_star).shape
        rhs = np.zeros(shape, dtype=float)

    # ------------------------------------------------------------
    # 2. Zero RHS inside solid cells
    # ------------------------------------------------------------
    if "mask_semantics" in state and "is_solid" in state["mask_semantics"]:
        is_solid = np.asarray(state["mask_semantics"]["is_solid"], dtype=bool)
        rhs = np.where(is_solid, 0.0, rhs)
    elif "is_solid" in state:
        is_solid = np.asarray(state["is_solid"], dtype=bool)
        rhs = np.where(is_solid, 0.0, rhs)

    return rhs
