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

    # Divergence operator (pure function)
    div_op = state["divergence"]["op"]

    # Compute divergence of predicted velocity
    div_u = div_op(U_star, V_star, W_star)

    rhs = (rho / dt) * div_u

    # Zero RHS in solid cells
    is_solid = np.asarray(state["mask_semantics"]["is_solid"], dtype=bool)
    rhs = np.array(rhs, copy=True)
    rhs[is_solid] = 0.0

    return rhs
