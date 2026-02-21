# src/step3/apply_boundary_conditions_post.py

import numpy as np
from src.step3.apply_domain_boundaries import apply_domain_boundaries

def apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new):
    """
    Step‑3 boundary‑condition reapplication.
    Ensures zero-velocity inside and on the surface of solid obstacles.
    """
    # Create the fields dictionary and apply Domain BCs (Inflow/Outflow/Slip) FIRST
    fields = {
        "U": np.array(U_new, copy=True),
        "V": np.array(V_new, copy=True),
        "W": np.array(W_new, copy=True),
        "P": P_new
    }
    
    # Apply Domain BCs (this might set an Inflow where a solid exists)
    fields = apply_domain_boundaries(state, fields)

    # Apply Solid Masks LAST (The Absolute Final Override)
    is_solid = ~state.is_fluid
    U, V, W = fields["U"], fields["V"], fields["W"]

    # --- U-velocity (Staggered in X: (nx+1, ny, nz)) ---
    # Internal faces: zeroed if either adjacent cell is solid
    U[1:-1, :, :][is_solid[:-1, :, :] | is_solid[1:, :, :]] = 0.0
    # Boundary faces: zeroed if the boundary cell is solid
    U[0, :, :][is_solid[0, :, :]] = 0.0    
    U[-1, :, :][is_solid[-1, :, :]] = 0.0  

    # --- V-velocity (Staggered in Y: (nx, ny+1, nz)) ---
    V[:, 1:-1, :][is_solid[:, :-1, :] | is_solid[:, 1:, :]] = 0.0
    V[:, 0, :][is_solid[:, 0, :]] = 0.0    
    V[:, -1, :][is_solid[:, -1, :]] = 0.0  

    # --- W-velocity (Staggered in Z: (nx, ny, nz+1)) ---
    W[:, :, 1:-1][is_solid[:, :, :-1] | is_solid[:, :, 1:]] = 0.0
    W[:, :, 0][is_solid[:, :, 0]] = 0.0    
    W[:, :, -1][is_solid[:, :, -1]] = 0.0  

    # Custom BC handler hook (if any)
    bc_handler = getattr(state, 'boundary_conditions', None)
    if callable(bc_handler):
        fields = bc_handler(state, fields)

    return fields