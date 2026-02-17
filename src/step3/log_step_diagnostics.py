# src/step3/log_step_diagnostics.py

import numpy as np


def log_step_diagnostics(state, fields, current_time, step_index):
    """
    Pure Step‑3 diagnostic logger.
    Produces a single diagnostic record without mutating state.
    """

    # ------------------------------------------------------------
    # 1. Health metrics from Step‑3
    # ------------------------------------------------------------
    div_norm = state.health.get("post_correction_divergence_norm", 0.0)
    max_vel = state.health.get("max_velocity_magnitude", 0.0)

    # PPE iteration count (optional; may not exist)
    ppe_iters = state.health.get("ppe_iterations", -1)

    # ------------------------------------------------------------
    # 2. Velocity fields
    # ------------------------------------------------------------
    U = np.asarray(fields["U"])
    V = np.asarray(fields["V"])
    W = np.asarray(fields["W"])

    rho = state.constants["rho"]

    # Kinetic energy per unit volume
    energy = 0.5 * rho * (np.mean(U**2) + np.mean(V**2) + np.mean(W**2))

    # ------------------------------------------------------------
    # 3. Return diagnostic record
    # ------------------------------------------------------------
    return {
        "time": float(current_time),
        "step_index": int(step_index),
        "divergence_norm": float(div_norm),
        "max_velocity": float(max_vel),
        "ppe_iterations": int(ppe_iters),
        "energy": float(energy),
    }
