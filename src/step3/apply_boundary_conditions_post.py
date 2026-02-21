import numpy as np
from src.step3.apply_domain_boundaries import apply_domain_boundaries

def apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new):
    """
    Step‑3 boundary‑condition reapplication.
    Takes corrected fields and returns new fields with BCs enforced.
    Ensures zero-velocity inside and on the surface of solid obstacles.
    """
    is_solid = ~state.is_fluid
    U = np.array(U_new, copy=True)
    V = np.array(V_new, copy=True)
    W = np.array(W_new, copy=True)

    # 1. Internal & Solid-Surface Masking
    # We explicitly handle boundaries and internal slices to avoid staggered IndexErrors.
    
    # --- U-velocity (Staggered in X) ---
    U[1:-1, :, :][is_solid[:-1, :, :] | is_solid[1:, :, :]] = 0.0
    U[0, :, :][is_solid[0, :, :]] = 0.0    # x_min boundary if solid
    U[-1, :, :][is_solid[-1, :, :]] = 0.0  # x_max boundary if solid

    # --- V-velocity (Staggered in Y) ---
    V[:, 1:-1, :][is_solid[:, :-1, :] | is_solid[:, 1:, :]] = 0.0
    V[:, 0, :][is_solid[:, 0, :]] = 0.0    # y_min boundary if solid
    V[:, -1, :][is_solid[:, -1, :]] = 0.0  # y_max boundary if solid

    # --- W-velocity (Staggered in Z) ---
    W[:, :, 1:-1][is_solid[:, :, :-1] | is_solid[:, :, 1:]] = 0.0
    W[:, :, 0][is_solid[:, :, 0]] = 0.0    # z_min boundary if solid
    W[:, :, -1][is_solid[:, :, -1]] = 0.0  # z_max boundary if solid

    fields = {"U": U, "V": V, "W": W, "P": P_new}

    # 2. Domain Enums: Apply x_min, x_max types (free-slip, no-slip) from config
    fields = apply_domain_boundaries(state, fields)

    # 3. Optional Custom BC handler (e.g., from unit tests or specialized physics)
    bc_handler = getattr(state, 'boundary_conditions', None)
    if callable(bc_handler):
        fields = bc_handler(state, fields)

    return fields