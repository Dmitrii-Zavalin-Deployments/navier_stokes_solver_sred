# src/step3/apply_boundary_conditions_pre.py

import numpy as np


def apply_boundary_conditions_pre(state, fields):
    """
    Pure Step‑3 boundary‑condition application BEFORE prediction.

    Inputs:
        state  – Step‑3 state dict (contains mask, is_fluid, is_solid)
        fields – dict with:
                 {
                     "U": ndarray,
                     "V": ndarray,
                     "W": ndarray,
                     "P": ndarray,
                 }

    Returns:
        new_fields – dict with the same keys, BCs applied, no mutation.
    """

    # ------------------------------------------------------------
    # FIX: Step‑3 schema uses "mask", not "mask_semantics"
    # ------------------------------------------------------------
    mask = np.asarray(state["mask"], dtype=int)
    is_solid = (mask == 0)

    U = np.array(fields["U"], copy=True)
    V = np.array(fields["V"], copy=True)
    W = np.array(fields["W"], copy=True)
    P = np.array(fields["P"], copy=True)

    nx, ny, nz = mask.shape

    # ------------------------------------------------------------
    # Zero-out velocity faces adjacent to solids (OR logic)
    # ------------------------------------------------------------

    # U faces: between i-1 and i
    solid_u = np.zeros_like(U, dtype=bool)
    solid_u[1:-1, :, :] = is_solid[:-1, :, :] | is_solid[1:, :, :]
    U[solid_u] = 0.0

    # V faces: between j-1 and j
    solid_v = np.zeros_like(V, dtype=bool)
    solid_v[:, 1:-1, :] = is_solid[:, :-1, :] | is_solid[:, 1:, :]
    V[solid_v] = 0.0

    # W faces: between k-1 and k
    solid_w = np.zeros_like(W, dtype=bool)
    solid_w[:, :, 1:-1] = is_solid[:, :, :-1] | is_solid[:, :, 1:]
    W[solid_w] = 0.0

    # ------------------------------------------------------------
    # Optional pure-function BC hook
    # ------------------------------------------------------------
    bc_fn = state.get("boundary_conditions_pre", None)

    if callable(bc_fn):
        U, V, W, P = bc_fn(state, U, V, W, P)

    return {
        "U": U,
        "V": V,
        "W": W,
        "P": P,
    }
