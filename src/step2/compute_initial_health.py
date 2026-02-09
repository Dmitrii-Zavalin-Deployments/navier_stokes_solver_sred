# src/step2/compute_initial_health.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .build_divergence_operator import build_divergence_operator


def _to_numpy(arr):
    return np.array(arr, dtype=float)


def compute_initial_health(state: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute initial diagnostics for solver stability.

    Metrics:
    - initial_divergence_norm: L2 norm of divergence(U, V, W)
    - max_velocity_magnitude: max |u| over the domain
    - cfl_advection_estimate: dt * (|u|/dx + |v|/dy + |w|/dz) max over domain
    """

    # Extract grid and constants
    grid = state["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    const = state["constants"]
    dt = float(const["dt"])
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

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

    # Compute divergence using the modern Step‑2 operator
    div_result = build_divergence_operator(state)
    div = _to_numpy(div_result["divergence"])

    # L2 norm of divergence
    initial_divergence_norm = float(np.linalg.norm(div.ravel(), ord=2))

    # Velocity magnitude at cell centers
    u_center = 0.5 * (U[0:nx, :, :] + U[1:nx + 1, :, :])
    v_center = 0.5 * (V[:, 0:ny, :] + V[:, 1:ny + 1, :])
    w_center = 0.5 * (W[:, :, 0:nz] + W[:, :, 1:nz + 1])

    vel_mag = np.sqrt(u_center**2 + v_center**2 + w_center**2)
    max_velocity_magnitude = float(np.max(vel_mag))

    # CFL estimate
    dx_safe = dx if dx > 0 else 1e-12
    dy_safe = dy if dy > 0 else 1e-12
    dz_safe = dz if dz > 0 else 1e-12

    cfl_field = dt * (
        np.abs(u_center) / dx_safe
        + np.abs(v_center) / dy_safe
        + np.abs(w_center) / dz_safe
    )

    cfl_advection_estimate = float(np.max(cfl_field))

    # Return pure JSON‑serializable health metrics
    return {
        "initial_divergence_norm": initial_divergence_norm,
        "max_velocity_magnitude": max_velocity_magnitude,
        "cfl_advection_estimate": cfl_advection_estimate,
    }
