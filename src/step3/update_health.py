# src/step3/update_health.py

import numpy as np


def _warn_pressure_velocity_coupling(state, fields):
    """
    Detects symptoms of pressure–velocity decoupling:
      • checkerboard pressure patterns
      • high divergence after projection
      • oscillatory velocity fields

    This is a soft diagnostic: it prints a warning but does not stop the solver.
    """

    # ------------------------------------------------------------
    # 1. Divergence threshold
    # ------------------------------------------------------------
    div = state.health.get("post_correction_divergence_norm", None)
    div_flag = div is not None and div > 1e-2

    # ------------------------------------------------------------
    # 2. Checkerboard pressure metric
    # ------------------------------------------------------------
    checkerboard = None
    P = fields.get("P", None)
    if P is not None and isinstance(P, np.ndarray) and P.ndim == 3:
        try:
            checkerboard = float(np.mean(np.abs(P[::2, ::2, ::2] - P[1::2, 1::2, 1::2])))
        except Exception:
            checkerboard = None

    checker_flag = checkerboard is not None and checkerboard > 1e-2

    # ------------------------------------------------------------
    # 3. Velocity oscillation metric
    # ------------------------------------------------------------
    vel_osc = None
    U = fields.get("U", None)
    if U is not None and isinstance(U, np.ndarray) and U.ndim == 3:
        try:
            vel_osc = float(np.mean(np.abs(U[:, 1:, :] - U[:, :-1, :])))
        except Exception:
            vel_osc = None

    vel_flag = vel_osc is not None and vel_osc > 1e-2

    # ------------------------------------------------------------
    # 4. Emit warning if any symptom is present
    # ------------------------------------------------------------
    if div_flag or checker_flag or vel_flag:
        print(
            "[WARNING] Potential pressure–velocity decoupling detected. "
            "Symptoms include checkerboard pressure, high divergence, or "
            "oscillatory velocity. Consider enabling Rhie–Chow stabilization."
        )


def update_health(state, fields, P_new):
    """
    Step‑3 health computation.
    Computes:
      • max_velocity_magnitude
      • post_correction_divergence_norm
      • cfl_advection_estimate

    Pure function: returns a new health dictionary.
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
    health = {
        "post_correction_divergence_norm": div_norm,
        "max_velocity_magnitude": max_vel,
        "cfl_advection_estimate": cfl,
    }

    # ------------------------------------------------------------
    # 6. Attach to state and run diagnostics
    # ------------------------------------------------------------
    state.health = health
    _warn_pressure_velocity_coupling(state, fields)

    return health
