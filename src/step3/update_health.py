# src/step3/update_health.py

import numpy as np


def update_health(state, fields, P_new):
    """
    Pure Step‑3 health computation.

    Computes:
      • max_velocity_magnitude  = max(|U|, |V|, |W|)
      • post_correction_divergence_norm
      • cfl_advection_estimate

    Robust to dummy states and missing operators.
    Does NOT mutate caller state.
    """

    # ------------------------------------------------------------
    # Extract velocity fields
    # ------------------------------------------------------------
    U = np.asarray(fields.get("U", np.zeros(1)))
    V = np.asarray(fields.get("V", np.zeros(1)))
    W = np.asarray(fields.get("W", np.zeros(1)))

    # ------------------------------------------------------------
    # 1. Max velocity magnitude (component‑wise)
    # ------------------------------------------------------------
    try:
        max_u = float(np.max(np.abs(U))) if U.size > 0 else 0.0
        max_v = float(np.max(np.abs(V))) if V.size > 0 else 0.0
        max_w = float(np.max(np.abs(W))) if W.size > 0 else 0.0
        max_vel = max(max_u, max_v, max_w)
    except Exception:
        max_vel = 0.0

    # ------------------------------------------------------------
    # 2. Divergence norm (post‑correction)
    # ------------------------------------------------------------
    div_norm = 0.0

    div_block = state.get("divergence", {})
    op = None

    if callable(div_block):
        op = div_block
    elif isinstance(div_block, dict) and callable(div_block.get("op")):
        op = div_block["op"]

    if op is not None:
        try:
            div = op(U, V, W)
            div_norm = float(np.linalg.norm(div))
        except Exception:
            div_norm = 0.0

    # ------------------------------------------------------------
    # 3. CFL estimate
    # ------------------------------------------------------------
    constants = state.get("constants", {})
    dt = float(constants.get("dt", 1.0))
    dx = float(constants.get("dx", 1.0))
    dy = float(constants.get("dy", 1.0))
    dz = float(constants.get("dz", 1.0))

    try:
        cfl = float(max_vel * dt / min(dx, dy, dz))
    except Exception:
        cfl = 0.0

    # ------------------------------------------------------------
    # Assemble health dictionary
    # ------------------------------------------------------------
    return {
        "post_correction_divergence_norm": div_norm,
        "max_velocity_magnitude": max_vel,
        "cfl_advection_estimate": cfl,
    }
