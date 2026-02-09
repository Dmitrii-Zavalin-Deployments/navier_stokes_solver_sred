# src/step2/build_laplacian_operators.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np


def _to_numpy(arr):
    return np.array(arr, dtype=float)


def _to_list(arr):
    return arr.tolist()


def _laplacian_scalar(F: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Standard 3D 7‑point Laplacian stencil with edge padding.
    """
    out = np.zeros_like(F)
    f = np.pad(F, ((1, 1), (1, 1), (1, 1)), mode="edge")

    out[:] = (
        (f[2:, 1:-1, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / (dx * dx)
        + (f[1:-1, 2:, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / (dy * dy)
        + (f[1:-1, 1:-1, 2:] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / (dz * dz)
    )
    return out


def build_laplacian_operators(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build Laplacian fields for U, V, W on a MAC grid.
    Input: Step‑1 state (dict with lists)
    Output: dict with laplacians (lists) and metadata.
    """

    const = state["constants"]
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    grid = state["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    # Canonical mask from Step‑1
    mask = _to_numpy(state["mask_3d"])
    is_fluid = (mask != 0)

    # Convert staggered fields to numpy
    U = _to_numpy(state["fields"]["U"])
    V = _to_numpy(state["fields"]["V"])
    W = _to_numpy(state["fields"]["W"])

    # Validate shapes
    if U.shape != (nx + 1, ny, nz):
        raise ValueError(f"U must have shape {(nx+1, ny, nz)}, got {U.shape}")

    if V.shape != (nx, ny + 1, nz):
        raise ValueError(f"V must have shape {(nx, ny+1, nz)}, got {V.shape}")

    if W.shape != (nx, ny, nz + 1):
        raise ValueError(f"W must have shape {(nx, ny, nz+1)}, got {W.shape}")

    # -----------------------------
    # Laplacian for U
    # -----------------------------
    lap_u = _laplacian_scalar(U, dx, dy, dz)

    fluid_u = np.zeros_like(U, bool)
    if U.shape[0] > 2:
        fluid_u[1:-1] = is_fluid[:-1] | is_fluid[1:]
    fluid_u[0] = is_fluid[0]
    fluid_u[-1] = is_fluid[-1]

    lap_u = np.where(fluid_u, lap_u, 0.0)

    # -----------------------------
    # Laplacian for V
    # -----------------------------
    lap_v = _laplacian_scalar(V, dx, dy, dz)

    fluid_v = np.zeros_like(V, bool)
    if V.shape[1] > 2:
        fluid_v[:, 1:-1] = is_fluid[:, :-1] | is_fluid[:, 1:]
    fluid_v[:, 0] = is_fluid[:, 0]
    fluid_v[:, -1] = is_fluid[:, -1]

    lap_v = np.where(fluid_v, lap_v, 0.0)

    # -----------------------------
    # Laplacian for W
    # -----------------------------
    lap_w = _laplacian_scalar(W, dx, dy, dz)

    fluid_w = np.zeros_like(W, bool)
    if W.shape[2] > 2:
        fluid_w[:, :, 1:-1] = is_fluid[:, :, :-1] | is_fluid[:, :, 1:]
    fluid_w[:, :, 0] = is_fluid[:, :, 0]
    fluid_w[:, :, -1] = is_fluid[:, :, -1]

    lap_w = np.where(fluid_w, lap_w, 0.0)

    # -----------------------------
    # Return JSON‑serializable output
    # -----------------------------
    return {
        "laplacians": {
            "u": _to_list(lap_u),
            "v": _to_list(lap_v),
            "w": _to_list(lap_w),
        },
        "laplacian_meta": {
            "stencil": "7‑point",
            "masking": "solid=0, fluid=1, boundary-fluid=-1",
        },
    }
