# src/step3/apply_boundary_conditions_post.py

import numpy as np
from src.step3.apply_domain_boundaries import apply_domain_boundaries

def apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new):
    """
    Step‑3 boundary‑condition reapplication.
    Ensures zero-velocity inside and on the surface of solid obstacles (No-Penetration).
    """
    fields = {
        "U": np.array(U_new, copy=True),
        "V": np.array(V_new, copy=True),
        "W": np.array(W_new, copy=True),
        "P": np.array(P_new, copy=True)
    }
    
    # Apply Domain BCs (Inflow/Outflow/Slip)
    fields = apply_domain_boundaries(state, fields)

    is_solid = ~state.is_fluid
    U, V, W = fields["U"], fields["V"], fields["W"]

    # --- U-velocity (nx+1, ny, nz) ---
    # Internal: If cell i or i-1 is solid, face i is 0
    U[1:-1, :, :][is_solid[:-1, :, :] | is_solid[1:, :, :]] = 0.0
    # Boundaries
    U[0, :, :][is_solid[0, :, :]] = 0.0    
    U[-1, :, :][is_solid[-1, :, :]] = 0.0  

    # --- V-velocity (nx, ny+1, nz) ---
    # Internal: If cell j or j-1 is solid, face j is 0
    V[:, 1:-1, :][is_solid[:, :-1, :] | is_solid[:, 1:, :]] = 0.0
    # Boundaries
    V[:, 0, :][is_solid[:, 0, :]] = 0.0    
    V[:, -1, :][is_solid[:, -1, :]] = 0.0  

    # --- W-velocity (nx, ny, nz+1) ---
    # Internal: If cell k or k-1 is solid, face k is 0
    W[:, :, 1:-1][is_solid[:, :, :-1] | is_solid[:, :, 1:]] = 0.0
    # Boundaries
    W[:, :, 0][is_solid[:, :, 0]] = 0.0    
    W[:, :, -1][is_solid[:, :, -1]] = 0.0  

    return fields