# src/step2/build_gradient_operators.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np


def _to_numpy(arr):
    return np.array(arr, dtype=float)


def _to_list(arr):
    return arr.tolist()


def build_gradient_operators(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build pressure gradient fields for a MAC grid.
    Input: Step‑1 state (dict with lists)
    Output: dict with gx, gy, gz (lists) and metadata.
    """

    grid = state["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    const = state["constants"]
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    # Canonical mask from Step‑1
    mask = _to_numpy(state["mask_3d"])
    is_fluid = (mask != 0)

    # Pressure field
    P = _to_numpy(state["fields"]["P"])

    # Validate P shape
    if P.shape != (nx, ny, nz):
        raise ValueError(f"P must have shape {(nx, ny, nz)}, got {P.shape}")

    # ------------------------------------------------------------------
    # ∂p/∂x at U faces: (nx+1, ny, nz)
    # ------------------------------------------------------------------
    gx = np.zeros((nx + 1, ny, nz), dtype=float)

    if nx > 0:
        gx[1:nx] = (P[1:] - P[:-1]) / dx
        gx[0] = 0.0
        gx[nx] = 0.0

    fluid_u = np.zeros_like(gx, bool)
    if nx > 0:
        fluid_u[1:nx] = is_fluid[:-1] & is_fluid[1:]
        fluid_u[0] = is_fluid[0]
        fluid_u[nx] = is_fluid[-1]

    gx = np.where(fluid_u, gx, 0.0)

    # ------------------------------------------------------------------
    # ∂p/∂y at V faces: (nx, ny+1, nz)
    # ------------------------------------------------------------------
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

    gy = np.where(fluid_v, gy, 0.0)

    # ------------------------------------------------------------------
    # ∂p/∂z at W faces: (nx, ny, nz+1)
    # ------------------------------------------------------------------
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

    gz = np.where(fluid_w, gz, 0.0)

    # ------------------------------------------------------------------
    # Return JSON‑serializable output
    # ------------------------------------------------------------------
    return {
        "pressure_gradients": {
            "gx": _to_list(gx),
            "gy": _to_list(gy),
            "gz": _to_list(gz),
        },
        "gradient_meta": {
            "stencil": "MAC",
            "masking": "solid=0, fluid=1, boundary-fluid=-1",
        },
    }
