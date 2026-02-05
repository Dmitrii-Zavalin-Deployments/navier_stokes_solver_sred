# file: step2/prepare_ppe_structure.py
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def prepare_ppe_structure(state: Any) -> Dict[str, Any]:
    """
    Prepare the structure and metadata for the Pressure Poisson Equation (PPE).

    This does NOT build the full sparse matrix A. Instead, it prepares:
    - rhs_builder: how to map divergence to RHS
    - solver_type, tolerance, max_iterations
    - ppe_is_singular: whether the PPE is structurally singular (no pressure Dirichlet)

    Parameters
    ----------
    state : Any
        SimulationState-like object with:
        - boundary_table: dict of faces -> list of BC dicts (from Step 1)
        - Constants: (rho, dt)

    Returns
    -------
    dict
        {
          "rhs_builder": callable,
          "solver_type": str,
          "tolerance": float,
          "max_iterations": int,
          "ppe_is_singular": bool,
        }
    """
    rho = float(state.Constants["rho"])
    dt = float(state.Constants["dt"])

    boundary_table = getattr(state, "boundary_table", {})

    def rhs_builder(divergence: np.ndarray) -> np.ndarray:
        """
        Map divergence field to PPE RHS.

        A common choice is:
          RHS = -rho / dt * divergence
        """
        return -rho / dt * divergence

    # Heuristic singularity detection:
    # If there is at least one pressure-type outlet/Dirichlet BC, we treat PPE as non-singular.
    has_pressure_dirichlet = False
    for face_bcs in boundary_table.values():
        for bc in face_bcs:
            role = bc.get("role")
            if role in ("outlet", "pressure_outlet", "pressure"):
                has_pressure_dirichlet = True
                break
        if has_pressure_dirichlet:
            break

    ppe_is_singular = not has_pressure_dirichlet

    ppe = {
        "rhs_builder": rhs_builder,
        "solver_type": "PCG",
        "tolerance": 1e-6,
        "max_iterations": 1000,
        "ppe_is_singular": ppe_is_singular,
    }

    state.PPE = ppe
    return ppe
