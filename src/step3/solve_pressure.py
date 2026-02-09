# src/step3/solve_pressure.py

import numpy as np


def solve_pressure(state, rhs_ppe):
    """
    Pure Step‑3 pressure solve.

    Solves:
        ∇² p = rhs

    Handles singularity by subtracting the mean over fluid cells.

    Inputs:
        state    – Step‑2 output dict
        rhs_ppe  – ndarray, RHS of PPE

    Returns:
        P_new          – pressure field
        ppe_metadata   – dict with solver diagnostics:
                         {
                             "converged": bool,
                             "last_iterations": int
                         }
    """

    ppe = state["ppe_structure"]
    solver = ppe.get("solver", None)
    is_singular = ppe.get("ppe_is_singular", False)

    # ------------------------------------------------------------
    # 1. Solve PPE
    # ------------------------------------------------------------
    if solver is None:
        # No solver provided → zero pressure
        P_new = np.zeros_like(rhs_ppe)
        metadata = {
            "converged": True,
            "last_iterations": 0,
        }
    else:
        # Solver returns (pressure, info_dict)
        P_new, info = solver(rhs_ppe)
        metadata = {
            "converged": info.get("converged", True),
            "last_iterations": info.get("iterations", -1),
        }

    # ------------------------------------------------------------
    # 2. Handle singularity (subtract mean over fluid cells)
    # ------------------------------------------------------------
    if is_singular:
        is_fluid = np.asarray(state["mask_semantics"]["is_fluid"], dtype=bool)
        if np.any(is_fluid):
            P_new = np.array(P_new, copy=True)
            P_new[is_fluid] -= P_new[is_fluid].mean()

    return P_new, metadata
