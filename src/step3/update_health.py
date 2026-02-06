# src/step3/update_health.py

import numpy as np

def update_health(state):
    """
    Compute divergence norm, max velocity, CFL.
    """

    div = state["Operators"]["divergence"]
    U, V, W = state["U"], state["V"], state["W"]

    div_u = div(U, V, W, state)
    fluid = state["is_fluid"]

    post_div = float(np.linalg.norm(div_u[fluid])) if np.any(fluid) else 0.0

    max_vel = float(max(np.max(np.abs(U)), np.max(np.abs(V)), np.max(np.abs(W))))

    dt = state["Constants"]["dt"]
    dx = state["Constants"]["dx"]
    dy = state["Constants"]["dy"]
    dz = state["Constants"]["dz"]

    cfl = dt * max_vel / min(dx, dy, dz)

    state["Health"] = {
        "post_correction_divergence_norm": post_div,
        "max_velocity_magnitude": max_vel,
        "cfl_advection_estimate": cfl,
    }
