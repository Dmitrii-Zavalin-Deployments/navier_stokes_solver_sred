# src/step3/build_ppe_rhs.py

import numpy as np


def build_ppe_rhs(state, U_star, V_star, W_star):
    """
    Step‑3 PPE RHS builder.
    Computes:
        rhs = (rho/dt) * divergence(U*, V*, W*)
    Pure function: does not mutate state.
    """

    rho = state.constants["rho"]
    dt = state.constants["dt"]

    # ------------------------------------------------------------
    # 1. Compute divergence using Step‑2 operator
    # ------------------------------------------------------------
    div_op = state.operators["divergence"]
    div = div_op(U_star, V_star, W_star)
    div = np.asarray(div, dtype=float)

    rhs = (rho / dt) * div

    # ------------------------------------------------------------
    # 2. Zero RHS inside solid cells
    # ------------------------------------------------------------
    is_solid = ~state.is_fluid
    rhs = np.where(is_solid, 0.0, rhs)

    return rhs
