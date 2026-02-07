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
    """

    # ------------------------------------------------------------------
    # Grid geometry
    # ------------------------------------------------------------------
    grid = state["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    # ------------------------------------------------------------------
    # Physical spacing (Step‑1 schema: constants.*)
    # ------------------------------------------------------------------
    const = state["constants"]
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    # ------------------------------------------------------------------
    # Mask: treat -1 (boundary-fluid) as fluid
    # ------------------------------------------------------------------
    mask = np.asarray(state["fields"]["Mask"])
    is_fluid = (mask != 0)

    # ------------------------------------------------------------------
    # Divergence operator
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Store operator in schema‑correct location
    # ------------------------------------------------------------------
    if "operators" not in state:
        state["operators"] = {}
    state["operators"]["divergence"] = divergence

    return divergence
