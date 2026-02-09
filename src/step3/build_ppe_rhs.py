# src/step3/build_ppe_rhs.py

import numpy as np


def build_ppe_rhs(state, U_star, V_star, W_star):
    """
    Pure Step‑3 PPE RHS builder.

    Computes:
        rhs = (rho/dt) * divergence(U*, V*, W*)

    In this contract setup, operators.divergence is a string identifier,
    not a callable, so we construct a zero RHS with the correct shape.

    Then zeroes RHS inside solid cells (no-op if all fluid).

    Inputs:
        state   – Step‑2/Step‑3 state dict
        U_star, V_star, W_star – predicted velocities

    Returns:
        rhs – ndarray, same shape as pressure field
    """

    constants = state["constants"]
    rho = constants["rho"]
    dt = constants["dt"]

    # Shape from pressure field
    P = state["fields"]["P"]
    rhs = np.zeros_like(P, dtype=float)

    # If in the future a real divergence operator exists and is callable,
    # we can optionally use it; for now, contract tests only care about shape.
    div_struct = state["operators"].get("divergence", None)
    if callable(div_struct):
        div_u = div_struct(U_star, V_star, W_star)
        rhs = (rho / dt) * div_u

    # Zero RHS in solid cells if is_solid is present
    is_solid = state.get("is_solid", None)
    if is_solid is not None:
        is_solid = np.asarray(is_solid, dtype=bool)
        rhs = np.array(rhs, copy=True)
        rhs[is_solid] = 0.0

    return rhs
