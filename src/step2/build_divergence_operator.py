# src/step2/build_divergence_operator.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np


def _to_numpy(arr):
    return np.asarray(arr, dtype=float)


def _to_list(arr):
    return arr.tolist()


def build_divergence_operator(state: Dict[str, Any]):
    """
    Return a callable divergence(U, V, W) operator.

    Tests expect:
        div_op = build_divergence_operator(state)
        div = div_op(U, V, W)

    NOT a dict. NOT precomputed divergence.
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

    def divergence(U, V, W):
        U = _to_numpy(U)
        V = _to_numpy(V)
        W = _to_numpy(W)

        # Validate shapes
        if U.shape != (nx + 1, ny, nz):
            raise ValueError(f"U must have shape {(nx+1, ny, nz)}, got {U.shape}")

        if V.shape != (nx, ny + 1, nz):
            raise ValueError(f"V must have shape {(nx, ny+1, nz)}, got {V.shape}")

        if W.shape != (nx, ny, nz + 1):
            raise ValueError(f"W must have shape {(nx, ny, nz+1)}, got {W.shape}")

        div = np.zeros((nx, ny, nz), float)

        # Standard MAC-grid divergence
        div += (U[1:, :, :] - U[:-1, :, :]) / dx
        div += (V[:, 1:, :] - V[:, :-1, :]) / dy
        div += (W[:, :, 1:] - W[:, :, :-1]) / dz

        # Zero out solid cells
        div[~is_fluid] = 0.0

        return div

    return divergence
