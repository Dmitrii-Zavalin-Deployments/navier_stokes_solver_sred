# src/step3/apply_boundary_conditions_pre.py

import numpy as np
from src.step3.apply_domain_boundaries import apply_domain_boundaries

def apply_boundary_conditions_pre(state, fields):
    """
    Step‑3 boundary‑condition application BEFORE prediction.
    Pure function: does not mutate state or input fields.
    
    Logic:
    1. Zero out faces adjacent to internal solids.
    2. Apply domain boundaries (inflow, etc.) from JSON config.
    """
    is_solid = ~state.is_fluid
    
    # 1. Copy fields (no mutation)
    U = np.array(fields["U"], copy=True)
    V = np.array(fields["V"], copy=True)
    W = np.array(fields["W"], copy=True)
    P = np.array(fields["P"], copy=True)

    # 2. Internal Mask: Zero-out velocity faces adjacent to solids
    # This prevents the predictor from starting with "junk" velocity inside walls
    U[1:-1, :, :][is_solid[:-1, :, :] | is_solid[1:, :, :]] = 0.0
    V[:, 1:-1, :][is_solid[:, :-1, :] | is_solid[:, 1:, :]] = 0.0
    W[:, :, 1:-1][is_solid[:, :, :-1] | is_solid[:, :, 1:]] = 0.0

    current_fields = {"U": U, "V": V, "W": W, "P": P}

    # 3. Domain Enums: Enforce the JSON schema boundaries (e.g., Inflow)
    # This ensures the prediction step respects the user's input values
    current_fields = apply_domain_boundaries(state, current_fields)

    # 4. Optional Custom BC hook
    bc_fn = state.boundary_conditions
    if callable(bc_fn):
        current_fields = bc_fn(state, current_fields)

    return current_fields