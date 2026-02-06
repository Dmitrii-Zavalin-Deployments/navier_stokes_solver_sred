# src/step3/solve_pressure.py

import numpy as np

def solve_pressure(state, rhs_ppe):
    """
    Solve ∇²p = rhs using PPE metadata.
    Handles singularity by subtracting mean over fluid cells.
    """

    ppe = state["PPE"]
    solver = ppe.get("solver", None)
    is_singular = ppe.get("ppe_is_singular", False)

    if solver is None:
        P_new = np.zeros_like(rhs_ppe)
        ppe["ppe_converged"] = True
    else:
        P_new, info = solver(rhs_ppe)
        ppe["ppe_converged"] = info.get("converged", True)
        ppe["last_iterations"] = info.get("iterations", -1)

    if is_singular:
        fluid = state["is_fluid"]
        if np.any(fluid):
            P_new[fluid] -= P_new[fluid].mean()

    return P_new
