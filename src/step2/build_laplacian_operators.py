# src/step2/build_laplacian_operators.py
from __future__ import annotations
import numpy as np
from src.solver_state import SolverState


def build_laplacian_operators(state: SolverState) -> None:
    """
    Build Laplacian operators for U, V, W.
    """

    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    dx2 = state.constants.inv_dx2
    dy2 = state.constants.inv_dy2
    dz2 = state.constants.inv_dz2
    is_fluid = state.is_fluid

    def lap(F):
        F = np.asarray(F)
        out = np.zeros_like(F)
        out[1:-1] += (F[2:] - 2 * F[1:-1] + F[:-2]) * dx2
        out[:, 1:-1] += (F[:, 2:] - 2 * F[:, 1:-1] + F[:, :-2]) * dy2
        out[:, :, 1:-1] += (F[:, :, 2:] - 2 * F[:, :, 1:-1] + F[:, :, :-2]) * dz2
        out[~is_fluid] = 0.0
        return out

    state.operators["laplacian_u"] = lap
    state.operators["laplacian_v"] = lap
    state.operators["laplacian_w"] = lap
