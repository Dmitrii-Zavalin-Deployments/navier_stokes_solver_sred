# src/step3/update_health.py

# file: src/step3/update_health.py

import numpy as np


def update_health(state, fields, P_new):
    """
    Step‑3 health computation.
    Computes:
      • max_velocity_magnitude
      • post_correction_divergence_norm
      • cfl_advection_estimate
    Pure function: does not mutate state.
    """

    # ------------------------------------------------------------
    # 1. Extract velocity fields
    # ------------------------------------------------------------
    U = np.asarray(fields["U"])
    V = np.asarray(fields["V"])
    W = np.asarray(fields["W"])

    # ------------------------------------------------------------
    # 2. Max velocity magnitude
    # ------------------------------------------------------------
    max_vel = float(
        max(
            np.max(np.abs(U)),
            np.max(np.abs(V)),
            np.max(np.abs(W)),
        )
    )

    # ------------------------------------------------------------
    # 3. Divergence norm (post‑correction)
    # ------------------------------------------------------------
    div_op = state.operators["divergence"]
    div = div_op(U, V, W)
    div_norm = float(np.linalg.norm(div))

    # ------------------------------------------------------------
    # 4. CFL estimate
    # ------------------------------------------------------------
    dt = state.constants["dt"]
    dx = state.constants["dx"]
    dy = state.constants["dy"]
    dz = state.constants["dz"]

    cfl = float(max_vel * dt / min(dx, dy, dz))

    # ------------------------------------------------------------
    # 5. Assemble health dictionary
    # ------------------------------------------------------------
    return {
        "post_correction_divergence_norm": div_norm,
        "max_velocity_magnitude": max_vel,
        "cfl_advection_estimate": cfl,
    }
