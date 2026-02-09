# src/step3/build_ppe_rhs.py

import numpy as np


def build_ppe_rhs(state, U_star, V_star, W_star):
    """
    Pure Step‑3 PPE RHS builder.

    Computes:
        rhs = (rho/dt) * divergence(U*, V*, W*)

    Then zeroes RHS inside solid cells.

    Inputs:
        state   – Step‑2 output dict
        U_star, V_star, W_star – predicted velocities

    Returns:
        rhs – ndarray, same shape as pressure field
    """

    constants = state["constants"]
    rho = constants["rho"]
    dt = constants["dt"]

    # ------------------------------------------------------------
    # Divergence operator (pure function)
    # Step‑2 stores it under: state["operators"]["divergence"]
    # ------------------------------------------------------------
    div_struct = state["operators"]["divergence"]

    # It may be:
    #   • a dict with {"op": callable}
    #   • a callable directly
    if isinstance(div_struct, dict) and callable(div_struct.get("op")):
        div_op = div_struct["op"]
    elif callable(div_struct):
        div_op = div_struct
    else:
        raise TypeError(
            "Divergence operator must be a callable or a dict containing an 'op' callable"
        )

    # ------------------------------------------------------------
    # Compute divergence of predicted velocity
    # ------------------------------------------------------------
    div_u = div_op(U_star, V_star, W_star)

    rhs = (rho / dt) * div_u

    # ------------------------------------------------------------
    # Zero RHS in solid cells
    # Step‑2 provides: state["is_solid"]
    # Step‑3 dummy states may not have it, so fallback to all-fluid
    # ------------------------------------------------------------
    is_solid = state.get("is_solid", None)

    if is_solid is not None:
        is_solid = np.asarray(is_solid, dtype=bool)
        rhs = np.array(rhs, copy=True)
        rhs[is_solid] = 0.0

    return rhs
