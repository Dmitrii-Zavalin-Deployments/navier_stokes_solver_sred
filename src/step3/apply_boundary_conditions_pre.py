# src/step3/apply_boundary_conditions_pre.py

import numpy as np

def apply_boundary_conditions_pre(state):
    """
    Apply velocity and pressure boundary conditions before prediction.
    Zero-out velocities in solid regions.
    """

    mask = state["Mask"]
    is_solid = (mask == 0)

    U, V, W = state["U"], state["V"], state["W"]
    nx, ny, nz = mask.shape

    # U faces
    solid_u = np.zeros_like(U, dtype=bool)
    solid_u[1:-1, :, :] = is_solid[:-1] & is_solid[1:]
    U[solid_u] = 0.0

    # V faces
    solid_v = np.zeros_like(V, dtype=bool)
    solid_v[:, 1:-1, :] = is_solid[:, :-1] & is_solid[:, 1:]
    V[solid_v] = 0.0

    # W faces
    solid_w = np.zeros_like(W, dtype=bool)
    solid_w[:, :, 1:-1] = is_solid[:, :, :-1] & is_solid[:, :, 1:]
    W[solid_w] = 0.0

    handler = state.get("BC_handler", None)
    if handler and hasattr(handler, "apply_pre"):
        handler.apply_pre(state)
