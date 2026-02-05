# src/step2/compute_initial_health.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np


def compute_initial_health(state: Any) -> Dict[str, float]:
    """
    Compute initial diagnostics for solver stability.

    Metrics:
    - initial_divergence_norm: L2 norm of divergence(U, V, W)
    - max_velocity_magnitude: max |u| over the domain
    - cfl_advection_estimate: dt * (|u|/dx + |v|/dy + |w|/dz) max over domain
    """

    # ------------------------------------------------------------
    # Ensure divergence operator exists (tests call this directly)
    # ------------------------------------------------------------
    if "Operators" not in state or "divergence" not in state["Operators"]:
        from .build_divergence_operator import build_divergence_operator
        build_divergence_operator(state)

    # ------------------------------------------------------------
    # Extract fields
    # ------------------------------------------------------------
    U = np.asarray(state.U)
    V = np.asarray(state.V)
    W = np.asarray(state.W)

    const = state["Constants"]
    dt = float(const["dt"])
    dx = float(const["dx"])
    dy = float(const["dy"])
    dz = float(const["dz"])

    # ------------------------------------------------------------
    # Divergence
    # ------------------------------------------------------------
    divergence_op = state["Operators"]["divergence"]
    div = divergence_op(U, V, W)

    initial_divergence_norm = float(np.linalg.norm(div.ravel(), ord=2))

    # ------------------------------------------------------------
    # Velocity magnitude at cell centers
    # ------------------------------------------------------------
    nx, ny, nz = div.shape

    u_center = 0.5 * (U[0:nx, :, :] + U[1:nx + 1, :, :])
    v_center = 0.5 * (V[:, 0:ny, :] + V[:, 1:ny + 1, :])
    w_center = 0.5 * (W[:, :, 0:nz] + W[:, :, 1:nz + 1])

    vel_mag = np.sqrt(u_center**2 + v_center**2 + w_center**2)
    max_velocity_magnitude = float(np.max(vel_mag))

    # ------------------------------------------------------------
    # CFL estimate
    # ------------------------------------------------------------
    dx_safe = dx if dx > 0 else 1e-12
    dy_safe = dy if dy > 0 else 1e-12
    dz_safe = dz if dz > 0 else 1e-12

    cfl_field = dt * (
        np.abs(u_center) / dx_safe
        + np.abs(v_center) / dy_safe
        + np.abs(w_center) / dz_safe
    )

    cfl_advection_estimate = float(np.max(cfl_field))

    # ------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------
    health = {
        "initial_divergence_norm": initial_divergence_norm,
        "max_velocity_magnitude": max_velocity_magnitude,
        "cfl_advection_estimate": cfl_advection_estimate,
    }

    state["Health"] = health
    return health
