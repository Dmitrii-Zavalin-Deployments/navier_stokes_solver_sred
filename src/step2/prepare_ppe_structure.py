# src/step2/prepare_ppe_structure.py
from __future__ import annotations
import numpy as np
from src.solver_state import SolverState


def prepare_ppe_structure(state: SolverState) -> None:
    """
    Prepare PPE metadata and RHS builder.
    """

    rho = state.constants.rho
    dt = state.constants.dt
    mask = np.asarray(state.mask)

    def rhs_builder(div):
        div = np.asarray(div)
        rhs = (rho / dt) * div
        rhs[mask == 0] = 0.0
        return rhs

    state.ppe = {
        "rhs_builder": rhs_builder,
        "solver_type": "PCG",
        "tolerance": 1e-6,
        "max_iterations": 1000,
        "ppe_is_singular": False,
    }
