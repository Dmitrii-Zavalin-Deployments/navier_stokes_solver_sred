# src/step3/apply_boundary_conditions_pre.py

import numpy as np


def apply_boundary_conditions_pre(state, fields):
    """
    Step‑3 boundary‑condition application BEFORE prediction.
    Pure function: does not mutate state or input fields.
    """

    # ------------------------------------------------------------
    # 1. Use Step‑2 semantics (is_fluid, is_boundary_cell)
    # ------------------------------------------------------------
    is_fluid = state.is_fluid
    is_solid = ~is_fluid

    # ------------------------------------------------------------
    # 2. Copy fields (no mutation)
    # ------------------------------------------------------------
    U = np.array(fields["U"], copy=True)
    V = np.array(fields["V"], copy=True)
    W = np.array(fields["W"], copy=True)
    P = np.array(fields["P"], copy=True)

    # ------------------------------------------------------------
    # 3. Zero-out velocity faces adjacent to solids
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
    # 4. Optional BC hook (pure function)
    # ------------------------------------------------------------
    bc_fn = state.boundary_conditions

    if callable(bc_fn):
        out = bc_fn(state, {"U": U, "V": V, "W": W, "P": P})
        U = out["U"]
        V = out["V"]
        W = out["W"]
        P = out["P"]

    return {"U": U, "V": V, "W": W, "P": P}
