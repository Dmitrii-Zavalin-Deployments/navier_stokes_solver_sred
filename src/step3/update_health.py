import numpy as np

def update_health(state, fields, P_new):
    """Step‑3 health computation using Sparse Operators on staggered grid."""
    U, V, W = fields["U"], fields["V"], fields["W"]
    dt = state.config.get("dt", state.constants.get("dt", 0.01))

    # 1. Max velocity magnitude (Absolute max across all staggered faces)
    max_vel = float(max(np.max(np.abs(U)), np.max(np.abs(V)), np.max(np.abs(W))))

    # 2. Divergence norm: div = D @ u_staggered
    div_op = state.operators["divergence"]
    velocity_vector = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    div_flat = div_op @ velocity_vector
    
    # Normalized by the number of fluid cells if available
    n_cells = np.sum(state.is_fluid) if state.is_fluid is not None else div_flat.size
    div_norm = float(np.linalg.norm(div_flat) / max(1, n_cells))

    # 3. CFL estimate (Hardened h_min)
    dx = state.constants.get("dx", 1.0)
    dy = state.constants.get("dy", 1.0)
    dz = state.constants.get("dz", 1.0)
    h_min = min(dx, dy, dz)
    cfl = float(max_vel * dt / h_min)

    health = {
        "post_correction_divergence_norm": div_norm,
        "max_velocity_magnitude": max_vel,
        "cfl_advection_estimate": cfl,
    }

    state.health = health
    # Optional diagnostic warning
    if div_norm > 1e-1:
        print(f"[HEALTH] Warning: High divergence detected ({div_norm:.2e})")
        
    return health