# src/step4/initialize_extended_fields.py

import numpy as np
from src.step4.allocate_extended_fields import allocate_extended_fields


def initialize_extended_fields(state):
    """
    Initialize extended staggered fields (P_ext, U_ext, V_ext, W_ext).

    Responsibilities:
    - Allocate extended fields via allocate_extended_fields.
    - Fill interior pressure and velocity:
        * Prefer Step-3 fields (state["fields"]) when present.
        * Fall back to initial_conditions otherwise.
    - Apply mask semantics:
        * mask == 0  → solid         → zero velocity, zero pressure.
        * mask == 1  → fluid         → normal interior.
        * mask == -1 → boundary-fluid → preserved (not zeroed here).
    - Expose extended arrays on legacy 'Domain' block for compatibility.
    """

    # ---------------------------------------------------------
    # 1. Allocate extended fields
    # ---------------------------------------------------------
    state = allocate_extended_fields(state)

    config = state.get("config", {})
    domain_cfg = config.get("domain", {})
    ic = config.get("initial_conditions", {})

    nx = domain_cfg["nx"]
    ny = domain_cfg["ny"]
    nz = domain_cfg["nz"]

    p0 = ic.get("initial_pressure", 0.0)
    u0, v0, w0 = ic.get("initial_velocity", [0.0, 0.0, 0.0])

    fields = state.get("fields", {})

    P_ext = state["P_ext"]
    U_ext = state["U_ext"]
    V_ext = state["V_ext"]
    W_ext = state["W_ext"]

    # ---------------------------------------------------------
    # 2. Normalize mask and semantics
    # ---------------------------------------------------------
    # Mask semantics (single source of truth):
    #   0   → solid
    #   1   → pure fluid
    #  -1   → boundary-fluid (fluid but special; not zeroed here)
    mask_raw = state.get("mask", None)
    mask = None
    if mask_raw is not None:
        mask = np.asarray(mask_raw, dtype=int)
        if mask.shape != (nx, ny, nz):
            # Try to broadcast if user provided a simpler shape
            try:
                mask = np.broadcast_to(mask, (nx, ny, nz))
            except ValueError as exc:
                raise RuntimeError(
                    f"[Step 4] Cannot broadcast mask of shape {mask.shape} "
                    f"to domain shape ({nx}, {ny}, {nz})"
                ) from exc

    # ---------------------------------------------------------
    # 3. Pressure interior: Step-3 P if available, else IC
    # ---------------------------------------------------------
    P_field = fields.get("P", None)
    if isinstance(P_field, np.ndarray) and P_field.shape == (nx, ny, nz):
        P_ext[1:nx+1, 1:ny+1, 1:nz+1] = P_field
    else:
        P_ext[1:nx+1, 1:ny+1, 1:nz+1] = p0

    # ---------------------------------------------------------
    # 4. Velocity interior: Step-3 fields override IC when present
    #    Staggering (MAC-style):
    #      U: (nx+1, ny,   nz)
    #      V: (nx,   ny+1, nz)
    #      W: (nx,   ny,   nz+1)
    # ---------------------------------------------------------
    U_field = fields.get("U", None)
    V_field = fields.get("V", None)
    W_field = fields.get("W", None)

    # U: interior slice (nx+1, ny, nz)
    if isinstance(U_field, np.ndarray) and U_field.shape == (nx+1, ny, nz):
        U_ext[1:nx+2, 1:ny+1, 1:nz+1] = U_field
    else:
        U_ext[1:nx+2, 1:ny+1, 1:nz+1] = u0

    # V: interior slice (nx, ny+1, nz)
    if isinstance(V_field, np.ndarray) and V_field.shape == (nx, ny+1, nz):
        V_ext[0:nx, 1:ny+2, 1:nz+1] = V_field
    else:
        V_ext[0:nx, 1:ny+2, 1:nz+1] = v0

    # W: interior slice (nx, ny, nz+1)
    if isinstance(W_field, np.ndarray) and W_field.shape == (nx, ny, nz+1):
        W_ext[0:nx, 0:ny, 1:nz+2] = W_field
    else:
        W_ext[0:nx, 0:ny, 1:nz+2] = w0

    # ---------------------------------------------------------
    # 5. Apply solid-mask zeroing (mask == 0)
    #    Boundary-fluid cells (mask == -1) are preserved.
    # ---------------------------------------------------------
    if mask is not None:
        solid = (mask == 0)

        # Pressure: 1-to-1 mapping
        P_ext[1:nx+1, 1:ny+1, 1:nz+1][solid] = 0.0

        # U faces: left and right faces of each solid cell
        U_ext[1:nx+1, 1:ny+1, 1:nz+1][solid] = 0.0   # left faces
        U_ext[2:nx+2, 1:ny+1, 1:nz+1][solid] = 0.0   # right faces

        # V faces: lower and upper faces in y
        V_ext[0:nx, 1:ny+1, 1:nz+1][solid] = 0.0     # lower faces
        V_ext[0:nx, 2:ny+2, 1:nz+1][solid] = 0.0     # upper faces

        # W faces: lower and upper faces in z
        W_ext[0:nx, 0:ny, 1:nz+1][solid] = 0.0       # lower faces
        W_ext[0:nx, 0:ny, 2:nz+2][solid] = 0.0       # upper faces

    # ---------------------------------------------------------
    # 6. Mark that initial velocity has been enforced
    # ---------------------------------------------------------
    bc_applied = state.get("BCApplied", {})
    bc_applied["initial_velocity_enforced"] = True
    state["BCApplied"] = bc_applied

    # ---------------------------------------------------------
    # 7. Ensure legacy Domain exposes extended arrays
    # ---------------------------------------------------------
    domain_legacy = state.get("Domain", {})
    domain_legacy["P_ext"] = P_ext
    domain_legacy["U_ext"] = U_ext
    domain_legacy["V_ext"] = V_ext
    domain_legacy["W_ext"] = W_ext
    state["Domain"] = domain_legacy

    return state
