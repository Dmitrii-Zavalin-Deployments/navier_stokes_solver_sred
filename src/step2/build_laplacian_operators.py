# src/step2/build_laplacian_operators.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np


def _to_numpy(arr):
    return np.asarray(arr, dtype=float)


def build_laplacian_operators(state: Dict[str, Any]):
    """
    Return (lap_u, lap_v, lap_w) callables.

    Tests expect:
        lap_u, lap_v, lap_w = build_laplacian_operators(state)
        out = lap_u(U)

    NOT a dict. NOT precomputed laplacians.
    """

    grid = state["grid"]
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])

    const = state["constants"]
    dx, dy, dz = float(const["dx"]), float(const["dy"]), float(const["dz"])

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 1.0 / (dz * dz)

    # Canonical mask from Step‑1
    mask = _to_numpy(state["mask_3d"])
    is_fluid = (mask != 0)

    # ----------------------------------------------------------------------
    # Laplacian for U (shape: nx+1, ny, nz)
    # ----------------------------------------------------------------------
    def lap_u(U):
        U = _to_numpy(U)

        if U.shape != (nx + 1, ny, nz):
            raise ValueError(f"U must have shape {(nx+1, ny, nz)}, got {U.shape}")

        out = np.zeros_like(U)

        # x-direction
        if nx > 0:
            out[1:-1] += (U[2:] - 2 * U[1:-1] + U[:-2]) * inv_dx2

        # y-direction
        if ny > 0:
            out[:, 1:-1] += (U[:, 2:] - 2 * U[:, 1:-1] + U[:, :-2]) * inv_dy2

        # z-direction
        if nz > 0:
            out[:, :, 1:-1] += (U[:, :, 2:] - 2 * U[:, :, 1:-1] + U[:, :, :-2]) * inv_dz2

        # Masking: U-face is fluid if either adjacent cell is fluid
        fluid_u = np.zeros_like(U, bool)
        if nx > 0:
            fluid_u[1:-1] = is_fluid[:-1] | is_fluid[1:]
        fluid_u[0] = is_fluid[0]
        fluid_u[-1] = is_fluid[-1]

        out[~fluid_u] = 0.0
        return out

    # ----------------------------------------------------------------------
    # Laplacian for V (shape: nx, ny+1, nz)
    # ----------------------------------------------------------------------
    def lap_v(V):
        V = _to_numpy(V)

        if V.shape != (nx, ny + 1, nz):
            raise ValueError(f"V must have shape {(nx, ny+1, nz)}, got {V.shape}")

        out = np.zeros_like(V)

        # x-direction
        if nx > 0:
            out[1:-1] += (V[2:] - 2 * V[1:-1] + V[:-2]) * inv_dx2

        # y-direction
        if ny > 0:
            out[:, 1:-1] += (V[:, 2:] - 2 * V[:, 1:-1] + V[:, :-2]) * inv_dy2

        # z-direction
        if nz > 0:
            out[:, :, 1:-1] += (V[:, :, 2:] - 2 * V[:, :, 1:-1] + V[:, :, :-2]) * inv_dz2

        # Masking: V-face is fluid if either adjacent cell is fluid
        fluid_v = np.zeros_like(V, bool)
        if ny > 0:
            fluid_v[:, 1:-1] = is_fluid[:, :-1] | is_fluid[:, 1:]
        fluid_v[:, 0] = is_fluid[:, 0]
        fluid_v[:, -1] = is_fluid[:, -1]

        out[~fluid_v] = 0.0
        return out

    # ----------------------------------------------------------------------
    # Laplacian for W (shape: nx, ny, nz+1)
    # ----------------------------------------------------------------------
    def lap_w(W):
        W = _to_numpy(W)

        if W.shape != (nx, ny, nz + 1):
            raise ValueError(f"W must have shape {(nx, ny, nz+1)}, got {W.shape}")

        out = np.zeros_like(W)

        # x-direction
        if nx > 0:
            out[1:-1] += (W[2:] - 2 * W[1:-1] + W[:-2]) * inv_dx2

        # y-direction
        if ny > 0:
            out[:, 1:-1] += (W[:, 2:] - 2 * W[:, 1:-1] + W[:, :-2]) * inv_dy2

        # z-direction
        if nz > 0:
            out[:, :, 1:-1] += (W[:, :, 2:] - 2 * W[:, :, 1:-1] + W[:, :, :-2]) * inv_dz2

        # Masking: W-face is fluid if either adjacent cell is fluid
        fluid_w = np.zeros_like(W, bool)
        if nz > 0:
            fluid_w[:, :, 1:-1] = is_fluid[:, :, :-1] | is_fluid[:, :, 1:]
        fluid_w[:, :, 0] = is_fluid[:, :, 0]
        fluid_w[:, :, -1] = is_fluid[:, :, -1]

        out[~fluid_w] = 0.0
        return out

    return lap_u, lap_v, lap_w
