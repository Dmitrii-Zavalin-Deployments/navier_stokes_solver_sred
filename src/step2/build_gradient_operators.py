# src/step2/build_gradient_operators.py
from __future__ import annotations

from typing import Any, Callable, Dict
import numpy as np


def build_gradient_operators(state: Any) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """
    Construct pressure gradient operators for each velocity component on a MAC grid.

    Mask-aware:
    - Gradients are computed only for fluid cells (mask != 0).
    - Solid cells (mask == 0) are set to zero.

    Parameters
    ----------
    state : Any
        SimulationState-like object with:
        - state["Grid"] = {"nx", "ny", "nz"}
        - state["Constants"] = {"dx", "dy", "dz"}
        - state["Mask"] = int[nx, ny, nz] with values {1, 0, -1}

    Returns
    -------
    dict
        {
          "gradient_p_x": callable,
          "gradient_p_y": callable,
          "gradient_p_z": callable,
        }
    """

    grid = state["Grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    const = state["Constants"]
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    mask = np.asarray(state["Mask"])
    is_fluid = (mask != 0)  # treat -1 as fluid

    # -------------------------------------------------------------
    # ∂p/∂x at U-locations: shape (nx+1, ny, nz)
    # -------------------------------------------------------------
    def gradient_p_x(P: np.ndarray) -> np.ndarray:
        gx = np.zeros((nx + 1, ny, nz), dtype=float)

        if nx > 0:
            # interior faces
            gx[1:nx, :, :] = (P[1:, :, :] - P[:-1, :, :]) / dx

            # boundaries (zero-gradient placeholder)
            gx[0, :, :] = 0.0
            gx[nx, :, :] = 0.0

        # Masking: zero out gradients adjacent to solid cells
        # A U-face is fluid if either adjacent cell is fluid
        fluid_u = np.zeros_like(gx, dtype=bool)
        if nx > 0:
            fluid_u[1:nx, :, :] = is_fluid[:-1, :, :] | is_fluid[1:, :, :]
            fluid_u[0, :, :] = is_fluid[0, :, :]
            fluid_u[nx, :, :] = is_fluid[-1, :, :]

        gx = np.where(fluid_u, gx, 0.0)
        return gx

    # -------------------------------------------------------------
    # ∂p/∂y at V-locations: shape (nx, ny+1, nz)
    # -------------------------------------------------------------
    def gradient_p_y(P: np.ndarray) -> np.ndarray:
        gy = np.zeros((nx, ny + 1, nz), dtype=float)

        if ny > 0:
            gy[:, 1:ny, :] = (P[:, 1:, :] - P[:, :-1, :]) / dy
            gy[:, 0, :] = 0.0
            gy[:, ny, :] = 0.0

        fluid_v = np.zeros_like(gy, dtype=bool)
        if ny > 0:
            fluid_v[:, 1:ny, :] = is_fluid[:, :-1, :] | is_fluid[:, 1:, :]
            fluid_v[:, 0, :] = is_fluid[:, 0, :]
            fluid_v[:, ny, :] = is_fluid[:, -1, :]

        gy = np.where(fluid_v, gy, 0.0)
        return gy

    # -------------------------------------------------------------
    # ∂p/∂z at W-locations: shape (nx, ny, nz+1)
    # -------------------------------------------------------------
    def gradient_p_z(P: np.ndarray) -> np.ndarray:
        gz = np.zeros((nx, ny, nz + 1), dtype=float)

        if nz > 0:
            gz[:, :, 1:nz] = (P[:, :, 1:] - P[:, :, :-1]) / dz
            gz[:, :, 0] = 0.0
            gz[:, :, nz] = 0.0

        fluid_w = np.zeros_like(gz, dtype=bool)
        if nz > 0:
            fluid_w[:, :, 1:nz] = is_fluid[:, :, :-1] | is_fluid[:, :, 1:]
            fluid_w[:, :, 0] = is_fluid[:, :, 0]
            fluid_w[:, :, nz] = is_fluid[:, :, -1]

        gz = np.where(fluid_w, gz, 0.0)
        return gz

    # -------------------------------------------------------------
    # Package operators
    # -------------------------------------------------------------
    ops = {
        "gradient_p_x": gradient_p_x,
        "gradient_p_y": gradient_p_y,
        "gradient_p_z": gradient_p_z,
    }

    if "Operators" not in state:
        state["Operators"] = {}
    state["Operators"].update(ops)

    return ops
