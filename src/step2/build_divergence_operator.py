# src/step2/build_divergence_operator.py
from __future__ import annotations
import numpy as np
from src.solver_state import SolverState


def build_divergence_operator(state: SolverState) -> None:
    """
    Construct MAC-grid divergence operator.
    Stores callable in state.operators["divergence"].
    """

    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    dx, dy, dz = state.constants.dx, state.constants.dy, state.constants.dz
    is_fluid = state.is_fluid

    def divergence(U, V, W):
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)

        div = (
            (U[1:, :, :] - U[:-1, :, :]) / dx
            + (V[:, 1:, :] - V[:, :-1, :]) / dy
            + (W[:, :, 1:] - W[:, :, :-1]) / dz
        )

        div[~is_fluid] = 0.0
        return div

    state.operators["divergence"] = divergence
