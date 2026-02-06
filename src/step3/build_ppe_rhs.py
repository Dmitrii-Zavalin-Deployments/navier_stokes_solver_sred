# src/step3/build_ppe_rhs.py


def build_ppe_rhs(state, U_star, V_star, W_star):
    """
    Build RHS of PPE: rhs = (rho/dt) * div(u*), zeroed in solid cells.
    """

    rho = state["Constants"]["rho"]
    dt = state["Constants"]["dt"]

    div = state["Operators"]["divergence"]
    rhs = (rho / dt) * div(U_star, V_star, W_star, state)

    rhs[state["Mask"] == 0] = 0.0
    return rhs
