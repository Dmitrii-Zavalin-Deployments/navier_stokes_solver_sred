# src/step2/build_gradient_operators.py
from __future__ import annotations

from typing import Any, Callable, Tuple
import numpy as np


def build_gradient_operators(state: Any) -> Tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    """
    Build ∂p/∂x, ∂p/∂y, ∂p/∂z operators on a MAC grid.
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
    is_fluid = (mask != 0)

    # -----------------------------
    # ∂p/∂x at U faces
    # -----------------------------
    def gradient_p_x(P: np.ndarray) -> np.ndarray:
        gx = np.zeros((nx + 1, ny, nz), dtype=float)

        if nx > 0:
            gx[1:nx] = (P[1:] - P[:-1]) / dx
            gx[0] = 0.0
            gx[nx] = 0.0

        fluid_u = np.zeros_like(gx, bool)
        if nx > 0:
            fluid_u[1:nx] = is_fluid[:-1] | is_fluid[1:]
            fluid_u[0] = is_fluid[0]
            fluid_u[nx] = is_fluid[-1]

        return np.where(fluid_u, gx, 0.0)

    # -----------------------------
    # ∂p/∂y at V faces
    # -----------------------------
    def gradient_p_y(P: np.ndarray) -> np.ndarray:
        gy = np.zeros((nx, ny + 1, nz), dtype=float)

        if ny > 0:
            gy[:, 1:ny] = (P[:, 1:] - P[:, :-1]) / dy
            gy[:, 0] = 0.0
            gy[:, ny] = 0.0

        fluid_v = np.zeros_like(gy, bool)
        if ny > 0:
            fluid_v[:, 1:ny] = is_fluid[:, :-1] | is_fluid[:, 1:]
            fluid_v[:, 0] = is_fluid[:, 0]
            fluid_v[:, ny] = is_fluid[:, -1]

        return np.where(fluid_v, gy, 0.0)

    # -----------------------------
    # ∂p/∂z at W faces
    # -----------------------------
    def gradient_p_z(P: np.ndarray) -> np.ndarray:
        gz = np.zeros((nx, ny, nz + 1), dtype=float)

        if nz > 0:
            gz[:, :, 1:nz] = (P[:, :, 1:] - P[:, :, :-1]) / dz
            gz[:, :, 0] = 0.0
            gz[:, :, nz] = 0.0

        fluid_w = np.zeros_like(gz, bool)
        if nz > 0:
            fluid_w[:, :, 1:nz] = is_fluid[:, :, :-1] | is_fluid[:, :, 1:]
            fluid_w[:, :, 0] = is_fluid[:, :, 0]
            fluid_w[:, :, nz] = is_fluid[:, :, -1]

        return np.where(fluid_w, gz, 0.0)

    # Store in state
    if "Operators" not in state:
        state["Operators"] = {}
    state["Operators"]["gradient_p_x"] = gradient_p_x
    state["Operators"]["gradient_p_y"] = gradient_p_y
    state["Operators"]["gradient_p_z"] = gradient_p_z

    # Return tuple (what tests expect)
    return gradient_p_x, gradient_p_y, gradient_p_z
