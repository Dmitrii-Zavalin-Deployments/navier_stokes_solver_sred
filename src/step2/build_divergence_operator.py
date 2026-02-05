# src/step2/build_divergence_operator.py
from __future__ import annotations

from typing import Any, Callable
import numpy as np


def build_divergence_operator(state: Any) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Construct the discrete divergence operator for staggered velocities.

    Uses a standard MAC-grid finite difference:
        div = dU/dx + dV/dy + dW/dz   at cell centers.

    Mask-aware:
    - Fluid cells (1) and boundary-fluid cells (-1) compute divergence normally.
    - Solid cells (0) are forced to zero.

    Parameters
    ----------
    state : Any
        SimulationState-like object with:
        - state["Grid"] = {"nx", "ny", "nz", "dx", "dy", "dz"}
        - state["Constants"] = {"dx", "dy", "dz", ...}
        - state["Mask"] = int[nx, ny, nz] with values {1, 0, -1}

    Returns
    -------
    divergence : callable
        divergence(U, V, W) -> np.ndarray[nx, ny, nz]
    """

    grid = state["Grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    const = state["Constants"]
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    # Mask: treat -1 (boundary-fluid) as fluid
    mask = np.asarray(state["Mask"])
    is_fluid = (mask != 0)

    def divergence(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute ∇·u at cell centers.

        U: (nx+1, ny,   nz)
        V: (nx,   ny+1, nz)
        W: (nx,   ny,   nz+1)
        """
        div = np.zeros((nx, ny, nz), dtype=float)

        # dU/dx
        if nx > 0:
            div += (U[1:, :, :] - U[:-1, :, :]) / dx

        # dV/dy
        if ny > 0:
            div += (V[:, 1:, :] - V[:, :-1, :]) / dy

        # dW/dz
        if nz > 0:
            div += (W[:, :, 1:] - W[:, :, :-1]) / dz

        # Zero out solid cells
        div = np.where(is_fluid, div, 0.0)
        return div

    # Store operator
    if "Operators" not in state:
        state["Operators"] = {}
    state["Operators"]["divergence"] = divergence

    return divergence
