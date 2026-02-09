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
        state  – Step‑2 output dict
        fields – dict with corrected velocity fields:
                 { "U": ndarray, "V": ndarray, "W": ndarray, "P": ndarray }
        P_new  – corrected pressure field (ndarray)

    Returns:
        health – dict:
        {
            "post_correction_divergence_norm": float,
            "max_velocity_magnitude": float,
            "cfl_advection_estimate": float
        }
    """

    # Divergence operator (pure)
    div_op = state["divergence"]["op"]

    U = np.asarray(fields["U"])
    V = np.asarray(fields["V"])
    W = np.asarray(fields["W"])

    # Compute divergence
    div_u = div_op(U, V, W)

    # Fluid mask
    is_fluid = np.asarray(state["mask_semantics"]["is_fluid"], dtype=bool)

    if np.any(is_fluid):
        post_div = float(np.linalg.norm(div_u[is_fluid]))
    else:
        post_div = 0.0

    # Max velocity magnitude
    max_vel = float(
        max(
            np.max(np.abs(U)),
            np.max(np.abs(V)),
            np.max(np.abs(W)),
        )
    )

    # CFL estimate
    constants = state["constants"]
    dt = constants["dt"]
    dx = constants["dx"]
    dy = constants["dy"]
    dz = constants["dz"]

    cfl = float(dt * max_vel / min(dx, dy, dz))

    return {
        "post_correction_divergence_norm": post_div,
        "max_velocity_magnitude": max_vel,
        "cfl_advection_estimate": cfl,
    }
