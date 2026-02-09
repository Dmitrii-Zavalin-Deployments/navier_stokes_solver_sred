# src/step3/log_step_diagnostics.py

import numpy as np


def log_step_diagnostics(state, fields, current_time, step_index):
    """
    Pure Step‑3 diagnostic logger.

    Inputs:
        state        – Step‑2 output dict
        fields       – dict with corrected velocity fields:
                       { "U": ndarray, "V": ndarray, "W": ndarray, "P": ndarray }
        current_time – float
        step_index   – int

    Returns:
        A dict representing one diagnostic record:
        {
            "time": float,
            "divergence_norm": float,
            "max_velocity": float,
            "ppe_iterations": int,
            "energy": float,
            "step_index": int
        }
    """

    # Health metrics from Step‑3
    health = state["health"]
    div_norm = health["post_correction_divergence_norm"]
    max_vel = health["max_velocity_magnitude"]

    # PPE metadata
    ppe = state.get("ppe_structure", {})
    ppe_iters = ppe.get("last_iterations", -1)

    # Velocity fields
    U = np.asarray(fields["U"])
    V = np.asarray(fields["V"])
    W = np.asarray(fields["W"])

    rho = state["constants"]["rho"]

    # Kinetic energy per unit volume
    energy = 0.5 * rho * (np.mean(U**2) + np.mean(V**2) + np.mean(W**2))

    return {
        "time": float(current_time),
        "step_index": int(step_index),
        "divergence_norm": float(div_norm),
        "max_velocity": float(max_vel),
        "ppe_iterations": int(ppe_iters),
        "energy": float(energy),
    }
