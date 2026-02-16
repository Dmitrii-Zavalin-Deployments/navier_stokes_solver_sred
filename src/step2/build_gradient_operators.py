# src/step2/build_gradient_operators.py
from __future__ import annotations
import numpy as np
from src.solver_state import SolverState


def build_gradient_operators(state: SolverState) -> None:
    """
    Build pressure gradient operators for MAC-grid velocities.
    """

    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    dx, dy, dz = state.constants.dx, state.constants.dy, state.constants.dz
    is_fluid = state.is_fluid

    def grad_x(P):
        P = np.asarray(P)
        gx = np.zeros((nx + 1, ny, nz))
        gx[1:nx] = (P[1:] - P[:-1]) / dx
        gx[~(is_fluid[:-1] & is_fluid[1:])] = 0.0
        return gx

    def grad_y(P):
        P = np.asarray(P)
        gy = np.zeros((nx, ny + 1, nz))
        gy[:, 1:ny] = (P[:, 1:] - P[:, :-1]) / dy
        gy[:, ~(is_fluid[:, :-1] & is_fluid[:, 1:])] = 0.0
        return gy

    def grad_z(P):
        P = np.asarray(P)
        gz = np.zeros((nx, ny, nz + 1))
        gz[:, :, 1:nz] = (P[:, :, 1:] - P[:, :, :-1]) / dz
        gz[:, :, ~(is_fluid[:, :, :-1] & is_fluid[:, :, 1:])] = 0.0
        return gz

    state.operators["gradient_p_x"] = grad_x
    state.operators["gradient_p_y"] = grad_y
    state.operators["gradient_p_z"] = grad_z
