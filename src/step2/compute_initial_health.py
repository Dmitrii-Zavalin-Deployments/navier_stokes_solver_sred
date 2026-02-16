# src/step2/compute_initial_health.py
from __future__ import_annotations
import numpy as np
from src.solver_state import SolverState


def compute_initial_health(state: SolverState) -> None:
    """
    Compute initial solver diagnostics.
    """

    U = np.asarray(state.fields["U"])
    V = np.asarray(state.fields["V"])
    W = np.asarray(state.fields["W"])

    div = state.operators["divergence"](U, V, W)
    div_norm = float(np.linalg.norm(div))

    vel_mag = np.sqrt(U**2 + V**2 + W**2)
    max_vel = float(np.max(vel_mag))

    dt = state.constants.dt
    dx = state.constants.dx
    dy = state.constants.dy
    dz = state.constants.dz

    cfl = float(np.max(dt * (np.abs(U)/dx + np.abs(V)/dy + np.abs(W)/dz)))

    state.health = {
        "initial_divergence_norm": div_norm,
        "max_velocity_magnitude": max_vel,
        "cfl_advection_estimate": cfl,
    }
