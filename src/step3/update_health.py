# src/step3/update_health.py

import numpy as np

def _warn_pressure_velocity_coupling(state, fields):
    """Detects symptoms of pressure–velocity decoupling."""
    div = state.health.get("post_correction_divergence_norm", None)
    div_flag = div is not None and div > 1e-2

    checker_flag = False
    P = fields.get("P", None)
    if P is not None and isinstance(P, np.ndarray) and P.ndim == 3:
        try:
            checkerboard = float(np.mean(np.abs(P[::2, ::2, ::2] - P[1::2, 1::2, 1::2])))
            checker_flag = checkerboard > 1e-2
        except: pass

    if div_flag or checker_flag:
        print("[WARNING] Potential pressure–velocity decoupling detected.")

def update_health(state, fields, P_new):
    """Step‑3 health computation using Sparse Operators."""
    U, V, W = fields["U"], fields["V"], fields["W"]

    # 1. Max velocity magnitude
    max_vel = float(max(np.max(np.abs(U)), np.max(np.abs(V)), np.max(np.abs(W))))

    # 2. Divergence norm: div = D @ u
    div_op = state.operators["divergence"]
    velocity_vector = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    div_flat = div_op @ velocity_vector
    div_norm = float(np.linalg.norm(div_flat) / max(1, div_flat.size))

    # 3. CFL estimate
    dt = state.constants["dt"]
    h_min = min(state.constants["dx"], state.constants["dy"], state.constants["dz"])
    cfl = float(max_vel * dt / h_min)

    health = {
        "post_correction_divergence_norm": div_norm,
        "max_velocity_magnitude": max_vel,
        "cfl_advection_estimate": cfl,
    }

    state.health = health
    _warn_pressure_velocity_coupling(state, fields)
    return health