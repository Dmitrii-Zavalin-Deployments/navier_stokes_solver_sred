# src/step2/build_laplacian_operators.py
from __future__ import annotations

from typing import Any, Callable, Tuple
import numpy as np


def _laplacian_scalar(F: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    out = np.zeros_like(F)
    f = np.pad(F, ((1, 1), (1, 1), (1, 1)), mode="edge")

    out[:] = (
        (f[2:, 1:-1, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / (dx * dx)
        + (f[1:-1, 2:, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / (dy * dy)
        + (f[1:-1, 1:-1, 2:] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / (dz * dz)
    )
    return out


def build_laplacian_operators(state: Any) -> Tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    # Physical spacing
    const = state["constants"]
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    # Mask: 0 = solid, ±1 = fluid/boundary-fluid
    mask = np.asarray(state["fields"]["Mask"])
    is_fluid = (mask != 0)

    # -----------------------------
    # Laplacian for U
    # -----------------------------
    def laplacian_u(U: np.ndarray) -> np.ndarray:
        lap = _laplacian_scalar(U, dx, dy, dz)

        nx_u = U.shape[0]
        fluid_u = np.zeros_like(U, bool)

        if nx_u > 2:
            fluid_u[1:-1] = is_fluid[:-1] | is_fluid[1:]
        fluid_u[0] = is_fluid[0]
        fluid_u[-1] = is_fluid[-1]

        return np.where(fluid_u, lap, 0.0)

    # -----------------------------
    # Laplacian for V
    # -----------------------------
    def laplacian_v(V: np.ndarray) -> np.ndarray:
        lap = _laplacian_scalar(V, dx, dy, dz)

        ny_v = V.shape[1]
        fluid_v = np.zeros_like(V, bool)

        if ny_v > 2:
            fluid_v[:, 1:-1] = is_fluid[:, :-1] | is_fluid[:, 1:]
        fluid_v[:, 0] = is_fluid[:, 0]
        fluid_v[:, -1] = is_fluid[:, -1]

        return np.where(fluid_v, lap, 0.0)

    # -----------------------------
    # Laplacian for W
    # -----------------------------
    def laplacian_w(W: np.ndarray) -> np.ndarray:
        lap = _laplacian_scalar(W, dx, dy, dz)

        nz_w = W.shape[2]
        fluid_w = np.zeros_like(W, bool)

        if nz_w > 2:
            fluid_w[:, :, 1:-1] = is_fluid[:, :, :-1] | is_fluid[:, :, 1:]
        fluid_w[:, :, 0] = is_fluid[:, :, 0]
        fluid_w[:, :, -1] = is_fluid[:, :, -1]

        return np.where(fluid_w, lap, 0.0)

    # Store in schema-correct location
    if "operators" not in state:
        state["operators"] = {}
    state["operators"]["laplacian_u"] = laplacian_u
    state["operators"]["laplacian_v"] = laplacian_v
    state["operators"]["laplacian_w"] = laplacian_w

    return laplacian_u, laplacian_v, laplacian_w