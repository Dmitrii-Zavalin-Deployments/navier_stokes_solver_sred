# src/step3/apply_boundary_conditions_post.py

import numpy as np
from src.step3.apply_domain_boundaries import apply_domain_boundaries

def apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new):
    """
    Step‑3 boundary‑condition reapplication.
    Takes corrected fields and returns new fields with BCs enforced.
    
    Logic Flow:
    1. Zero out internal solid faces (Internal Mask).
    2. Apply domain-specific Enums (JSON Contract).
    3. Call custom BC handler if exists.
    """
    is_solid = ~state.is_fluid
    U = np.array(U_new, copy=True)
    V = np.array(V_new, copy=True)
    W = np.array(W_new, copy=True)

    # 1. Internal Mask: Zero faces adjacent to solid cells
    # This ensures no-leakage for obstacles
    U[1:-1, :, :][is_solid[:-1, :, :] | is_solid[1:, :, :]] = 0.0
    V[:, 1:-1, :][is_solid[:, :-1, :] | is_solid[:, 1:, :]] = 0.0
    W[:, :, 1:-1][is_solid[:, :, :-1] | is_solid[:, :, 1:] = 0.0

    fields = {"U": U, "V": V, "W": W, "P": P_new}

    # 2. Domain Enums: Apply x_min, x_max, etc. from JSON config
    fields = apply_domain_boundaries(state, fields)

    # 3. Optional Custom BC handler
    bc_handler = state.boundary_conditions
    if callable(bc_handler):
        fields = bc_handler(state, fields)

    return fields