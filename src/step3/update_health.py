# src/step3/update_health.py

import numpy as np


def update_health(state, fields, P_new):
    """
    Pure Step‑3 health computation.

    Computes:
        • post-correction divergence norm
        • max velocity magnitude
        • CFL estimate

    Inputs:
        state  – Step‑2/Step‑3 state dict
        fields – dict with corrected velocity fields:
                 { "U": ndarray, "V": ndarray, "W": ndarray, "P": ndarray }
        P_new  – corrected pressure field (ndarray or (ndarray, meta))

    Returns:
        health – dict:
        {
            "post_correction_divergence_norm": float,
            "max_velocity_magnitude": float,
            "cfl_advection_estimate": float
        }
    """

    const = state["constants"]
    dt = const.get("dt", 1.0)
    dx = const.get("dx", 1.0)
    dy = const.get("dy", 1.0)
    dz = const.get("dz", 1.0)

    # ------------------------------------------------------------
    # If velocities are missing (dummy Step‑3 state), return zeros
    # ------------------------------------------------------------
    if not all(k in fields for k in ("U", "V", "W")):
        return {
            "post_correction_divergence_norm": 0.0,
            "max_velocity_magnitude": 0.0,
            "cfl_advection_estimate": 0.0,
        }

    # ------------------------------------------------------------
    # Extract velocity fields (real run)
    # ------------------------------------------------------------
    U = np.asarray(fields["U"])
    V = np.asarray(fields["V"])
    W = np.asarray(fields["W"])

    # ------------------------------------------------------------
    # 1. Divergence operator (may be missing)
    # ------------------------------------------------------------
    div_norm = 0.0
    ops = state.get("operators", {})
    div_struct = ops.get("divergence", None)

    div_op = None
    if isinstance(div_struct, dict) and callable(div_struct.get("op")):
        div_op = div_struct["op"]
    elif callable(div_struct):
        div_op = div_struct

    if div_op is not None:
        div = div_op(U, V, W)
        div_norm = float(np.linalg.norm(div))

    # ------------------------------------------------------------
    # 2. Max velocity magnitude
    # ------------------------------------------------------------
    vel_mag = np.sqrt(U**2 + V**2 + W**2)
    max_vel = float(np.max(vel_mag)) if vel_mag.size > 0 else 0.0

    # ------------------------------------------------------------
    # 3. CFL estimate
    # ------------------------------------------------------------
    cfl = float(max_vel * dt / min(dx, dy, dz))

    return {
        "post_correction_divergence_norm": div_norm,
        "max_velocity_magnitude": max_vel,
        "cfl_advection_estimate": cfl,
    }
