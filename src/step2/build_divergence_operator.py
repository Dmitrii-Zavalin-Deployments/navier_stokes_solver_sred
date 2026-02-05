# file: step2/build_divergence_operator.py
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def build_divergence_operator(state: Any) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Construct the discrete divergence operator for staggered velocities.

    Uses a standard MAC-grid finite difference:
      div = dU/dx + dV/dy + dW/dz at cell centers.

    Mask-aware:
    - Divergence is computed only on fluid cells.
    - Solid cells are set to zero.

    Parameters
    ----------
    state : Any
        SimulationState-like object with:
        - Grid (nx, ny, nz)
        - Constants (dx, dy, dz)
        - is_fluid (bool[nx, ny, nz])

    Returns
    -------
    divergence : callable
        divergence(U, V, W) -> np.ndarray[nx, ny, nz]
    """
    grid = state.Grid
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    dx = float(state.Constants["dx"])
    dy = float(state.Constants["dy"])
    dz = float(state.Constants["dz"])

    is_fluid = np.asarray(state.is_fluid)

    def divergence(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute ∇·u at cell centers.

        U: (nx+1, ny,   nz)
        V: (nx,   ny+1, nz)
        W: (nx,   ny,   nz+1)
        """
        div = np.zeros((nx, ny, nz), dtype=float)

        # dU/dx
        div += (U[1:, :, :] - U[:-1, :, :]) / dx

        # dV/dy
        div += (V[:, 1:, :] - V[:, :-1, :]) / dy

        # dW/dz
        div += (W[:, :, 1:] - W[:, :, :-1]) / dz

        # Mask: only fluid cells are meaningful; solid cells set to zero.
        div = np.where(is_fluid, div, 0.0)
        return div

    state.Operators = getattr(state, "Operators", {})
    state.Operators["divergence"] = divergence
    return divergence
