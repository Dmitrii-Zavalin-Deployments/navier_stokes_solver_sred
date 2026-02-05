# src/step2/build_gradient_operators.py
from __future__ import annotations

from typing import Any, Callable, Tuple
import numpy as np


def build_gradient_operators(
    state: Any,
) -> Tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    """
    Construct pressure gradient operators for each velocity component on a MAC grid.

    Returns three callables:
      grad_x(P), grad_y(P), grad_z(P)
    """

    grid = state["Grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    const = state["Constants"]
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    # Mask semantics: 0 = solid, ±1 = fluid / boundary-fluid
    mask = np.asarray(state["Mask"])
    is_fluid = (mask != 0)

    # ------------------------------------------------------------------
    # ∂p/∂x at U faces: shape (nx+1, ny, nz)
    # Face i sits between cell i-1 and i.
    # We treat a face as fluid ONLY if both adjacent cells are fluid-like.
    # ------------------------------------------------------------------
    def gradient_p_x(P: np.ndarray) -> np.ndarray:
        gx = np.zeros((nx + 1, ny, nz), dtype=float)

        if nx > 0:
            gx[1:nx] = (P[1:] - P[:-1]) / dx
            gx[0] = 0.0
            gx[nx] = 0.0

        fluid_u = np.zeros_like(gx, bool)
        if nx > 0:
            # interior faces: between i-1 and i
            fluid_u[1:nx] = is_fluid[:-1] & is_fluid[1:]
            # boundary faces: require the single adjacent cell to be fluid
            fluid_u[0] = is_fluid[0]
            fluid_u[nx] = is_fluid[-1]

        return np.where(fluid_u, gx, 0.0)

    # ------------------------------------------------------------------
    # ∂p/∂y at V faces: shape (nx, ny+1, nz)
    # Face j sits between cell j-1 and j.
    # Again, require both adjacent cells to be fluid-like.
    # ------------------------------------------------------------------
    def gradient_p_y(P: np.ndarray) -> np.ndarray:
        gy = np.zeros((nx, ny + 1, nz), dtype=float)

        if ny > 0:
            gy[:, 1:ny] = (P[:, 1:] - P[:, :-1]) / dy
            gy[:, 0] = 0.0
            gy[:, ny] = 0.0

        fluid_v = np.zeros_like(gy, bool)
        if ny > 0:
            fluid_v[:, 1:ny] = is_fluid[:, :-1] & is_fluid[:, 1:]
            fluid_v[:, 0] = is_fluid[:, 0]
            fluid_v[:, ny] = is_fluid[:, -1]

        return np.where(fluid_v, gy, 0.0)

    # ------------------------------------------------------------------
    # ∂p/∂z at W faces: shape (nx, ny, nz+1)
    # Face k sits between cell k-1 and k.
    # Require both adjacent cells to be fluid-like.
    # ------------------------------------------------------------------
    def gradient_p_z(P: np.ndarray) -> np.ndarray:
        gz = np.zeros((nx, ny, nz + 1), dtype=float)

        if nz > 0:
            gz[:, :, 1:nz] = (P[:, :, 1:] - P[:, :, :-1]) / dz
            gz[:, :, 0] = 0.0
            gz[:, :, nz] = 0.0

        fluid_w = np.zeros_like(gz, bool)
        if nz > 0:
            fluid_w[:, :, 1:nz] = is_fluid[:, :, :-1] & is_fluid[:, :, 1:]
            fluid_w[:, :, 0] = is_fluid[:, :, 0]
            fluid_w[:, :, nz] = is_fluid[:, :, -1]

        return np.where(fluid_w, gz, 0.0)

    # Store in state for later use
    if "Operators" not in state:
        state["Operators"] = {}
    state["Operators"]["gradient_p_x"] = gradient_p_x
    state["Operators"]["gradient_p_y"] = gradient_p_y
    state["Operators"]["gradient_p_z"] = gradient_p_z

    # Tests expect a tuple of callables
    return gradient_p_x, gradient_p_y, gradient_p_z
