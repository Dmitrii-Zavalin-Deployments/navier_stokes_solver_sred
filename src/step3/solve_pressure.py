# src/step3/solve_pressure.py

import numpy as np


def solve_pressure(state, rhs_ppe):
    """
    Step‑3 pressure solve.
    Solves the PPE:
        ∇² p = rhs
    using the solver configuration provided in Step‑2.
    Pure function: does not mutate state.
    """

    ppe = state.ppe
    solver_type = ppe["solver_type"]
    tolerance = ppe["tolerance"]
    max_iter = ppe["max_iterations"]
    is_singular = ppe["ppe_is_singular"]

    # ------------------------------------------------------------
    # 1. Select solver implementation
    # ------------------------------------------------------------
    if solver_type == "SOR":
        solver = _solve_pressure_sor
    elif solver_type == "PCG":
        solver = _solve_pressure_pcg
    else:
        raise ValueError(f"Unknown PPE solver_type: {solver_type}")

    # ------------------------------------------------------------
    # 2. Solve PPE
    # ------------------------------------------------------------
    P_new, iterations = solver(
        state,
        rhs_ppe,
        tolerance=tolerance,
        max_iterations=max_iter,
    )

    # ------------------------------------------------------------
    # 3. Handle singularity (subtract mean over fluid cells)
    # ------------------------------------------------------------
    if is_singular:
        is_fluid = state.is_fluid
        if np.any(is_fluid):
            P_new = np.array(P_new, copy=True)
            P_new[is_fluid] -= P_new[is_fluid].mean()

    # ------------------------------------------------------------
    # 4. Return pressure + metadata
    # ------------------------------------------------------------
    metadata = {
        "converged": iterations < max_iter,
        "last_iterations": iterations,
    }

    return P_new, metadata


# ----------------------------------------------------------------------
# Internal solver implementations
# ----------------------------------------------------------------------

def _solve_pressure_sor(state, rhs, tolerance, max_iterations):
    """
    Simple SOR PPE solver.
    Placeholder implementation — replace with your real SOR.
    """
    P = np.zeros_like(rhs)
    # TODO: real SOR implementation
    return P, 0


def _solve_pressure_pcg(state, rhs, tolerance, max_iterations):
    """
    Simple PCG PPE solver.
    Placeholder implementation — replace with your real PCG.
    """
    P = np.zeros_like(rhs)
    # TODO: real PCG implementation
    return P, 0
