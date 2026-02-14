# src/step4/initialize_staggered_fields.py

import numpy as np
from src.step4.allocate_extended_fields import allocate_extended_fields


def _normalize_mask(mask_raw, nx, ny, nz):
    """
    Normalize mask to shape (nx, ny, nz) by broadcasting or tiling.
    """
    mask = np.asarray(mask_raw, dtype=int)

    # Already correct
    if mask.shape == (nx, ny, nz):
        return mask

    # Try broadcasting
    try:
        return np.broadcast_to(mask, (nx, ny, nz)).copy()
    except ValueError:
        pass

    # Fallback: tile along dimensions
    if mask.ndim != 3:
        raise ValueError(f"Mask must be 3D or broadcastable; got {mask.shape}")

    mx, my, mz = mask.shape
    tx = nx // mx if mx != nx else 1
    ty = ny // my if my != ny else 1
    tz = nz // mz if mz != nz else 1

    tiled = np.tile(mask, (tx, ty, tz))

    # Final broadcast if needed
    if tiled.shape != (nx, ny, nz):
        tiled = np.broadcast_to(tiled, (nx, ny, nz)).copy()

    return tiled


def initialize_staggered_fields(state):
    """
    Initialize extended staggered fields using Step‑3 fields when available,
    falling back to initial_conditions only when fields are missing.
    """

    # ---------------------------------------------------------
    # 1. Allocate extended fields
    # ---------------------------------------------------------
    state = allocate_extended_fields(state)

    config = state.get("config", {})
    ic = config.get("initial_conditions", {})
    p0 = ic.get("initial_pressure", 0.0)
    u0, v0, w0 = ic.get("initial_velocity", [0.0, 0.0, 0.0])

    nx = config["domain"]["nx"]
    ny = config["domain"]["ny"]
    nz = config["domain"]["nz"]

    # Normalize mask
    mask_raw = state.get("mask", None)
    if mask_raw is not None:
        mask = _normalize_mask(mask_raw, nx, ny, nz)
    else:
        mask = None

    fields = state.get("fields", {})

    P_ext = state["P_ext"]
    U_ext = state["U_ext"]
    V_ext = state["V_ext"]
    W_ext = state["W_ext"]

    # ---------------------------------------------------------
    # 2. Pressure interior: Step‑3 P if available, else IC
    # ---------------------------------------------------------
    P_field = fields.get("P")
    if isinstance(P_field, np.ndarray) and P_field.shape == (nx, ny, nz):
        P_ext[1:nx+1, 1:ny+1, 1:nz+1] = P_field
    else:
        P_ext[1:nx+1, 1:ny+1, 1:nz+1] = p0

    # ---------------------------------------------------------
    # 3. Velocity interior: Step‑3 U/V/W if available, else IC
    # ---------------------------------------------------------
    U_field = fields.get("U")
    V_field = fields.get("V")
    W_field = fields.get("W")

    # U staggered in x → interior (nx+1, ny, nz)
    if isinstance(U_field, np.ndarray) and U_field.shape == (nx+1, ny, nz):
        U_ext[1:nx+2, 1:ny+1, 1:nz+1] = U_field
    else:
        U_ext[1:nx+2, 1:ny+1, 1:nz+1] = u0

    # V staggered in y → interior (nx, ny+1, nz)
    if isinstance(V_field, np.ndarray) and V_field.shape == (nx, ny+1, nz):
        V_ext[0:nx, 1:ny+2, 1:nz+1] = V_field
    else:
        V_ext[0:nx, 1:ny+2, 1:nz+1] = v0

    # W staggered in z → interior (nx, ny, nz+1)
    if isinstance(W_field, np.ndarray) and W_field.shape == (nx, ny, nz+1):
        W_ext[0:nx, 0:ny, 1:nz+2] = W_field
    else:
        W_ext[0:nx, 0:ny, 1:nz+2] = w0

    # ---------------------------------------------------------
    # 4. Solid-mask zeroing (mask == 0)
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
    # 5. Boundary-fluid preservation (mask == -1)
    # ---------------------------------------------------------
    if mask is not None:
        boundary_fluid = (mask == -1)

        P_ext[1:nx+1, 1:ny+1, 1:nz+1][boundary_fluid] = P_ext[
            1:nx+1, 1:ny+1, 1:nz+1
        ][boundary_fluid]

        U_ext[1:nx+1, 1:ny+1, 1:nz+1][boundary_fluid] = U_ext[
            1:nx+1, 1:ny+1, 1:nz+1
        ][boundary_fluid]
        U_ext[2:nx+2, 1:ny+1, 1:nz+1][boundary_fluid] = U_ext[
            2:nx+2, 1:ny+1, 1:nz+1
        ][boundary_fluid]

        V_ext[0:nx, 1:ny+1, 1:nz+1][boundary_fluid] = V_ext[
            0:nx, 1:ny+1, 1:nz+1
        ][boundary_fluid]
        V_ext[0:nx, 2:ny+2, 1:nz+1][boundary_fluid] = V_ext[
            0:nx, 2:ny+2, 1:nz+1
        ][boundary_fluid]

        W_ext[0:nx, 0:ny, 1:nz+1][boundary_fluid] = W_ext[
            0:nx, 0:ny, 1:nz+1
        ][boundary_fluid]
        W_ext[0:nx, 0:ny, 2:nz+2][boundary_fluid] = W_ext[
            0:nx, 0:ny, 2:nz+2
        ][boundary_fluid]

    # ---------------------------------------------------------
    # 6. BC vs mask conflict rule: solid mask wins
    # ---------------------------------------------------------
    if mask is not None:
        solid = (mask == 0)

        U_ext[1:nx+1, 1:ny+1, 1:nz+1][solid] = 0.0
        U_ext[2:nx+2, 1:ny+1, 1:nz+1][solid] = 0.0

        V_ext[0:nx, 1:ny+1, 1:nz+1][solid] = 0.0
        V_ext[0:nx, 2:ny+2, 1:nz+1][solid] = 0.0

        W_ext[0:nx, 0:ny, 1:nz+1][solid] = 0.0
        W_ext[0:nx, 0:ny, 2:nz+2][solid] = 0.0

    # ---------------------------------------------------------
    # 7. Mark that initial velocity has been enforced
    # ---------------------------------------------------------
    bc_applied = state.get("BCApplied", {})
    bc_applied["initial_velocity_enforced"] = True
    state["BCApplied"] = bc_applied

    # ---------------------------------------------------------
    # 8. Legacy Domain exposure
    # ---------------------------------------------------------
    domain_legacy = state.get("Domain", {})
    domain_legacy["P_ext"] = P_ext
    domain_legacy["U_ext"] = U_ext
    domain_legacy["V_ext"] = V_ext
    domain_legacy["W_ext"] = W_ext
    state["Domain"] = domain_legacy

    return state
