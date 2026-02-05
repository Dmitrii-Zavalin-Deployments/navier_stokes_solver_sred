# src/step2/build_laplacian_operators.py
from __future__ import annotations

from typing import Any, Callable, Dict
import numpy as np


def _laplacian_scalar(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Simple 3D 7-point Laplacian for a scalar field on a regular grid.
    Neumann-like behavior at boundaries (copy edge values).
    """
    out = np.zeros_like(field, dtype=float)

    # Pad with edge values to handle boundaries
    f = np.pad(field, ((1, 1), (1, 1), (1, 1)), mode="edge")

    dx2 = dx * dx
    dy2 = dy * dy
    dz2 = dz * dz

    out[:, :, :] = (
        (f[2:, 1:-1, 1:-1] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / dx2
        + (f[1:-1, 2:, 1:-1] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / dy2
        + (f[1:-1, 1:-1, 2:] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / dz2
    )

    return out


def build_laplacian_operators(state: Any) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """
    Construct Laplacian operators for viscous diffusion and PPE-related terms.

    Mask-aware:
    - Laplacian is computed everywhere, then zeroed out on non-fluid cells.

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
          "laplacian_u": callable,
          "laplacian_v": callable,
          "laplacian_w": callable,
        }
    """

    const = state["Constants"]
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    mask = np.asarray(state["Mask"])
    is_fluid = (mask != 0)  # treat -1 as fluid

    # -------------------------------------------------------------
    # Laplacian for U (nx+1, ny, nz)
    # -------------------------------------------------------------
    def laplacian_u(U: np.ndarray) -> np.ndarray:
        lap = _laplacian_scalar(U, dx, dy, dz)

        # Map cell-centered mask to U-faces
        nx_u, ny_u, nz_u = U.shape
        fluid_u = np.zeros_like(U, dtype=bool)

        # interior faces
        if nx_u > 2:
            fluid_u[1:-1, :, :] = is_fluid[:-1, :, :] | is_fluid[1:, :, :]

        # left boundary
        fluid_u[0, :, :] = is_fluid[0, :, :]

        # right boundary
        fluid_u[-1, :, :] = is_fluid[-1, :, :]

        return np.where(fluid_u, lap, 0.0)

    # -------------------------------------------------------------
    # Laplacian for V (nx, ny+1, nz)
    # -------------------------------------------------------------
    def laplacian_v(V: np.ndarray) -> np.ndarray:
        lap = _laplacian_scalar(V, dx, dy, dz)

        nx_v, ny_v, nz_v = V.shape
        fluid_v = np.zeros_like(V, dtype=bool)

        if ny_v > 2:
            fluid_v[:, 1:-1, :] = is_fluid[:, :-1, :] | is_fluid[:, 1:, :]

        fluid_v[:, 0, :] = is_fluid[:, 0, :]
        fluid_v[:, -1, :] = is_fluid[:, -1, :]

        return np.where(fluid_v, lap, 0.0)

    # -------------------------------------------------------------
    # Laplacian for W (nx, ny, nz+1)
    # -------------------------------------------------------------
    def laplacian_w(W: np.ndarray) -> np.ndarray:
        lap = _laplacian_scalar(W, dx, dy, dz)

        nx_w, ny_w, nz_w = W.shape
        fluid_w = np.zeros_like(W, dtype=bool)

        if nz_w > 2:
            fluid_w[:, :, 1:-1] = is_fluid[:, :, :-1] | is_fluid[:, :, 1:]

        fluid_w[:, :, 0] = is_fluid[:, :, 0]
        fluid_w[:, :, -1] = is_fluid[:, :, -1]

        return np.where(fluid_w, lap, 0.0)

    # -------------------------------------------------------------
    # Package operators
    # -------------------------------------------------------------
    ops = {
        "laplacian_u": laplacian_u,
        "laplacian_v": laplacian_v,
        "laplacian_w": laplacian_w,
    }

    if "Operators" not in state:
        state["Operators"] = {}
    state["Operators"].update(ops)

    return ops
