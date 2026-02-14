# src/step4/initialize_extended_fields.py

import numpy as np


def initialize_extended_fields(state):
    """
    Initialize extended staggered fields (P_ext, U_ext, V_ext, W_ext).

    Responsibilities:
    - Allocate extended fields (inlined here).
    - Fill interior pressure and velocity:
        * Prefer Step-3 fields (state["fields"]) when present.
        * Fall back to initial_conditions otherwise.
    - Apply mask semantics:
        * mask == 0  → solid          → zero velocity, zero pressure.
        * mask == 1  → pure fluid     → normal interior.
        * mask == -1 → boundary-fluid → preserved (not zeroed here).
    - Expose extended arrays on legacy 'Domain' block for compatibility.
    """

    config = state.get("config", {})
    domain_cfg = config.get("domain", {})
    ic = config.get("initial_conditions", {})

    nx = domain_cfg["nx"]
    ny = domain_cfg["ny"]
    nz = domain_cfg["nz"]

    # ---------------------------------------------------------
    # 1. Allocate extended fields (inlined)
    # ---------------------------------------------------------
    state["P_ext"] = np.zeros((nx + 2, ny + 2, nz + 2), dtype=float)
    state["U_ext"] = np.zeros((nx + 3, ny + 2, nz + 2), dtype=float)
    state["V_ext"] = np.zeros((nx + 2, ny + 3, nz + 2), dtype=float)
    state["W_ext"] = np.zeros((nx + 2, ny + 2, nz + 3), dtype=float)

    P_ext = state["P_ext"]
    U_ext = state["U_ext"]
    V_ext = state["V_ext"]
    W_ext = state["W_ext"]

    p0 = ic.get("initial_pressure", 0.0)
    u0, v0, w0 = ic.get("initial_velocity", [0.0, 0.0, 0.0])

    fields = state.get("fields", {})

    # ---------------------------------------------------------
    # 2. Normalize mask and semantics
    # ---------------------------------------------------------
    mask_raw = state.get("mask", None)
    mask = None
    if mask_raw is not None:
        if isinstance(mask_raw, np.ndarray):
            mask = mask_raw.astype(int)
        else:
            mask = np.array(mask_raw, dtype=int)

        if mask.shape != (nx, ny, nz):
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
    # ---------------------------------------------------------
    if mask is not None:
        solid = (mask == 0)

        P_ext[1:nx+1, 1:ny+1, 1:nz+1][solid] = 0.0

        U_ext[1:nx+1, 1:ny+1, 1:nz+1][solid] = 0.0
        U_ext[2:nx+2, 1:ny+1, 1:nz+1][solid] = 0.0

        V_ext[0:nx, 1:ny+1, 1:nz+1][solid] = 0.0
        V_ext[0:nx, 2:ny+2, 1:nz+1][solid] = 0.0

        W_ext[0:nx, 0:ny, 1:nz+1][solid] = 0.0
        W_ext[0:nx, 0:ny, 2:nz+2][solid] = 0.0

    # ---------------------------------------------------------
    # 6. Mark that initial velocity has been enforced
    # ---------------------------------------------------------
    bc_applied = state.get("BCApplied", {})
    bc_applied["initial_velocity_enforced"] = True
    state["BCApplied"] = bc_applied

    # ---------------------------------------------------------
    # 7. Expose extended arrays on legacy Domain block
    # ---------------------------------------------------------
    domain_legacy = state.get("Domain", {})
    domain_legacy["P_ext"] = P_ext
    domain_legacy["U_ext"] = U_ext
    domain_legacy["V_ext"] = V_ext
    domain_legacy["W_ext"] = W_ext
    state["Domain"] = domain_legacy

    return state
