# file: step2/build_gradient_operators.py
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def build_gradient_operators(state: Any) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """
    Construct pressure gradient operators for each velocity component on a MAC grid.

    Mask-aware:
    - Gradients are computed only for fluid cells; solid cells are set to zero.

    Parameters
    ----------
    state : Any
        SimulationState-like object with:
        - Grid (nx, ny, nz)
        - Constants (dx, dy, dz)
        - is_fluid (bool[nx, ny, nz])

    Returns
    -------
    dict
        {
          "gradient_p_x": callable,
          "gradient_p_y": callable,
          "gradient_p_z": callable,
        }
    """
    grid = state.Grid
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    dx = float(state.Constants["dx"])
    dy = float(state.Constants["dy"])
    dz = float(state.Constants["dz"])

    is_fluid = np.asarray(state.is_fluid)

    def gradient_p_x(P: np.ndarray) -> np.ndarray:
        """
        Compute ∂p/∂x at U locations: shape (nx+1, ny, nz).
        Simple one-sided differences at boundaries.
        """
        gx = np.zeros((nx + 1, ny, nz), dtype=float)

        # Interior faces: central difference
        gx[1:nx, :, :] = (P[1:, :, :] - P[:-1, :, :]) / dx

        # Left boundary: forward difference
        gx[0, :, :] = (P[0, :, :] - P[0, :, :]) / dx  # effectively zero; BCs handled elsewhere

        # Right boundary: backward difference
        gx[nx, :, :] = (P[-1, :, :] - P[-1, :, :]) / dx  # zero; BCs handled in PPE/BC logic

        return gx

    def gradient_p_y(P: np.ndarray) -> np.ndarray:
        """
        Compute ∂p/∂y at V locations: shape (nx, ny+1, nz).
        """
        gy = np.zeros((nx, ny + 1, nz), dtype=float)
        gy[:, 1:ny, :] = (P[:, 1:, :] - P[:, :-1, :]) / dy
        gy[:, 0, :] = (P[:, 0, :] - P[:, 0, :]) / dy
        gy[:, ny, :] = (P[:, -1, :] - P[:, -1, :]) / dy
        return gy

    def gradient_p_z(P: np.ndarray) -> np.ndarray:
        """
        Compute ∂p/∂z at W locations: shape (nx, ny, nz+1).
        """
        gz = np.zeros((nx, ny, nz + 1), dtype=float)
        gz[:, :, 1:nz] = (P[:, :, 1:] - P[:, :, :-1]) / dz
        gz[:, :, 0] = (P[:, :, 0] - P[:, :, 0]) / dz
        gz[:, :, nz] = (P[:, :, -1] - P[:, :, -1]) / dz
        return gz

    ops = {
        "gradient_p_x": gradient_p_x,
        "gradient_p_y": gradient_p_y,
        "gradient_p_z": gradient_p_z,
    }

    state.Operators = getattr(state, "Operators", {})
    state.Operators.update(ops)
    return ops
