# file: step2/compute_initial_health.py
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

    Parameters
    ----------
    state : Any
        SimulationState-like object with:
        - U, V, W
        - Constants (dt, dx, dy, dz)
        - Operators["divergence"]

    Returns
    -------
    dict
        {
          "initial_divergence_norm": float,
          "max_velocity_magnitude": float,
          "cfl_advection_estimate": float,
        }
    """
    U = np.asarray(state.U)
    V = np.asarray(state.V)
    W = np.asarray(state.W)

    dt = float(state.Constants["dt"])
    dx = float(state.Constants["dx"])
    dy = float(state.Constants["dy"])
    dz = float(state.Constants["dz"])

    divergence_op = state.Operators["divergence"]
    div = divergence_op(U, V, W)

    initial_divergence_norm = float(np.linalg.norm(div.ravel(), ord=2))

    # Compute velocity magnitude at cell centers by simple averaging to P-grid.
    # First, build cell-centered velocities by averaging neighboring faces.
    nx, ny, nz = div.shape

    u_center = 0.5 * (U[0:nx, :, :] + U[1:nx + 1, :, :])
    v_center = 0.5 * (V[:, 0:ny, :] + V[:, 1:ny + 1, :])
    w_center = 0.5 * (W[:, :, 0:nz] + W[:, :, 1:nz + 1])

    vel_mag = np.sqrt(u_center**2 + v_center**2 + w_center**2)
    max_velocity_magnitude = float(np.max(vel_mag))

    # Conservative 3D CFL estimate: dt * (|u|/dx + |v|/dy + |w|/dz)
    cfl_field = dt * (
        np.abs(u_center) / dx
        + np.abs(v_center) / dy
        + np.abs(w_center) / dz
    )
    cfl_advection_estimate = float(np.max(cfl_field))

    health = {
        "initial_divergence_norm": initial_divergence_norm,
        "max_velocity_magnitude": max_velocity_magnitude,
        "cfl_advection_estimate": cfl_advection_estimate,
    }

    state.Health = health
    return health
