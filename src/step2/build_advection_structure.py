# src/step2/build_advection_structure.py
from __future__ import annotations
import numpy as np
from src.solver_state import SolverState


def build_advection_structure(state: SolverState) -> None:
    """
    Build advection operators for U, V, W.
    """

    dx, dy, dz = state.constants.dx, state.constants.dy, state.constants.dz
    is_fluid = state.is_fluid

    def adv(F):
        F = np.asarray(F)
        out = np.zeros_like(F)
        out[1:-1] += F[1:-1] * (F[2:] - F[:-2]) / (2 * dx)
        out[:, 1:-1] += F[:, 1:-1] * (F[:, 2:] - F[:, :-2]) / (2 * dy)
        out[:, :, 1:-1] += F[:, :, 1:-1] * (F[:, :, 2:] - F[:, :, :-2]) / (2 * dz)
        out[~is_fluid] = 0.0
        return out

    state.operators["advection_u"] = adv
    state.operators["advection_v"] = adv
    state.operators["advection_w"] = adv
