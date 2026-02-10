# src/step2/build_gradient_operators.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np


def _to_numpy(arr):
    return np.asarray(arr, dtype=float)


def _to_list(arr):
    return arr.tolist()


def build_gradient_operators(state: Dict[str, Any]):
    """
    Return (grad_x, grad_y, grad_z) callables.

    Tests expect:
        grad_x, grad_y, grad_z = build_gradient_operators(state)
        gx = grad_x(P)
        gy = grad_y(P)
        gz = grad_z(P)

    NOT a dict. NOT precomputed gradients.
    """

    grid = state["grid"]
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])

    const = state["constants"]
    dx, dy, dz = float(const["dx"]), float(const["dy"]), float(const["dz"])

    # Canonical mask from Step‑1
    mask = _to_numpy(state["mask_3d"])
    is_fluid = (mask != 0)

    # ----------------------------------------------------------------------
    # ∂p/∂x at U faces → shape (nx+1, ny, nz)
    # ----------------------------------------------------------------------
    def grad_x(P):
        P = _to_numpy(P)

        if P.shape != (nx, ny, nz):
            raise ValueError(f"P must have shape {(nx, ny, nz)}, got {P.shape}")

        gx = np.zeros((nx + 1, ny, nz), float)

        if nx > 0:
            gx[1:nx] = (P[1:] - P[:-1]) / dx

        # Masking: U-face is fluid only if both adjacent cells are fluid
        fluid_u = np.zeros_like(gx, bool)
        if nx > 0:
            fluid_u[1:nx] = is_fluid[:-1] & is_fluid[1:]
        fluid_u[0] = is_fluid[0]
        fluid_u[nx] = is_fluid[-1]

        gx[~fluid_u] = 0.0
        return gx

    # ----------------------------------------------------------------------
    # ∂p/∂y at V faces → shape (nx, ny+1, nz)
    # ----------------------------------------------------------------------
    def grad_y(P):
        P = _to_numpy(P)

        if P.shape != (nx, ny, nz):
            raise ValueError(f"P must have shape {(nx, ny, nz)}, got {P.shape}")

        gy = np.zeros((nx, ny + 1, nz), float)

        if ny > 0:
            gy[:, 1:ny] = (P[:, 1:] - P[:, :-1]) / dy

        fluid_v = np.zeros_like(gy, bool)
        if ny > 0:
            fluid_v[:, 1:ny] = is_fluid[:, :-1] & is_fluid[:, 1:]
        fluid_v[:, 0] = is_fluid[:, 0]
        fluid_v[:, ny] = is_fluid[:, -1]

        gy[~fluid_v] = 0.0
        return gy

    # ----------------------------------------------------------------------
    # ∂p/∂z at W faces → shape (nx, ny, nz+1)
    # ----------------------------------------------------------------------
    def grad_z(P):
        P = _to_numpy(P)

        if P.shape != (nx, ny, nz):
            raise ValueError(f"P must have shape {(nx, ny, nz)}, got {P.shape}")

        gz = np.zeros((nx, ny, nz + 1), float)

        if nz > 0:
            gz[:, :, 1:nz] = (P[:, :, 1:] - P[:, :, :-1]) / dz

        fluid_w = np.zeros_like(gz, bool)
        if nz > 0:
            fluid_w[:, :, 1:nz] = is_fluid[:, :, :-1] & is_fluid[:, :, 1:]
        fluid_w[:, :, 0] = is_fluid[:, :, 0]
        fluid_w[:, :, nz] = is_fluid[:, :, -1]

        gz[~fluid_w] = 0.0
        return gz

    return grad_x, grad_y, grad_z
