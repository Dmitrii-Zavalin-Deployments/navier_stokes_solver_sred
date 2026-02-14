# src/step4/initialize_staggered_fields.py

import numpy as np
from src.step4.allocate_extended_fields import allocate_extended_fields


def initialize_staggered_fields(state):
    """
    Initialize the extended staggered fields (U_ext, V_ext, W_ext, P_ext)
    using the fields produced by Step 3 (and falling back to initial
    conditions where needed).

    Responsibilities:
    - Allocate extended fields (via allocate_extended_fields).
    - Fill interior pressure and velocity from Step‑3 fields when available.
    - Fall back to initial_conditions if fields are missing.
    - Apply solid-mask zeroing (mask == 0).
    - Preserve boundary-fluid cells (mask == -1).
    - Apply BC vs mask conflict rule (solid mask wins).
    """

    # ---------------------------------------------------------
    # 1. Allocate extended fields (P_ext, U_ext, V_ext, W_ext)
    # ---------------------------------------------------------
    state = allocate_extended_fields(state)

    config = state.get("config", {})
    ic = config.get("initial_conditions", {})
    p0 = ic.get("initial_pressure", 0.0)
    u0, v0, w0 = ic.get("initial_velocity", [0.0, 0.0, 0.0])

    nx = config["domain"]["nx"]
    ny = config["domain"]["ny"]
    nz = config["domain"]["nz"]

    mask_raw = state.get("mask", None)
    mask = np.asarray(mask_raw) if mask_raw is not None else None

    fields = state.get("fields", {})

    P_ext = state["P_ext"]
    U_ext = state["U_ext"]
    V_ext = state["V_ext"]
    W_ext = state["W_ext"]

    # ---------------------------------------------------------
    # 2. Fill interior pressure
    #    Prefer Step‑3 P field; fall back to p0 if absent.
    # ---------------------------------------------------------
    P_field = fields.get("P", None)
    if isinstance(P_field, np.ndarray) and P_field.shape == (nx, ny, nz):
        P_ext[1:nx+1, 1:ny+1, 1:nz+1] = P_field
    else:
        P_ext[1:nx+1, 1:ny+1, 1:nz+1] = p0

    # ---------------------------------------------------------
    # 3. Fill interior velocities (respect staggering)
    #    Prefer Step‑3 U/V/W fields; fall back to u0/v0/w0.
    # ---------------------------------------------------------
    U_field = fields.get("U", None)
    V_field = fields.get("V", None)
    W_field = fields.get("W", None)

    # U staggered in x → interior shape (nx+1, ny, nz)
    if isinstance(U_field, np.ndarray) and U_field.shape == (nx+1, ny, nz):
        U_ext[1:nx+2, 1:ny+1, 1:nz+1] = U_field
    else:
        U_ext[1:nx+2, 1:ny+1, 1:nz+1] = u0

    # V staggered in y → interior shape (nx, ny+1, nz)
    if isinstance(V_field, np.ndarray) and V_field.shape == (nx, ny+1, nz):
        V_ext[0:nx, 1:ny+2, 1:nz+1] = V_field
    else:
        V_ext[0:nx, 1:ny+2, 1:nz+1] = v0

    # W staggered in z → interior shape (nx, ny, nz+1)
    if isinstance(W_field, np.ndarray) and W_field.shape == (nx, ny, nz+1):
        W_ext[0:nx, 0:ny, 1:nz+2] = W_field
    else:
        W_ext[0:nx, 0:ny, 1:nz+2] = w0

    # ---------------------------------------------------------
    # 4. Apply solid-mask zeroing (mask == 0)
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
    # 5. Boundary-fluid preservation (mask == -1)
    # ---------------------------------------------------------
    if mask is not None:
        boundary_fluid = (mask == -1)

        # Pressure
        P_ext[1:nx+1, 1:ny+1, 1:nz+1][boundary_fluid] = P_ext[
            1:nx+1, 1:ny+1, 1:nz+1
        ][boundary_fluid]

        # U faces
        U_ext[1:nx+1, 1:ny+1, 1:nz+1][boundary_fluid] = U_ext[
            1:nx+1, 1:ny+1, 1:nz+1
        ][boundary_fluid]
        U_ext[2:nx+2, 1:ny+1, 1:nz+1][boundary_fluid] = U_ext[
            2:nx+2, 1:ny+1, 1:nz+1
        ][boundary_fluid]

        # V faces
        V_ext[0:nx, 1:ny+1, 1:nz+1][boundary_fluid] = V_ext[
            0:nx, 1:ny+1, 1:nz+1
        ][boundary_fluid]
        V_ext[0:nx, 2:ny+2, 1:nz+1][boundary_fluid] = V_ext[
            0:nx, 2:ny+2, 1:nz+1
        ][boundary_fluid]

        # W faces
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
    # 8. Ensure legacy Domain exposes extended arrays
    # ---------------------------------------------------------
    domain_legacy = state.get("Domain", {})
    domain_legacy["P_ext"] = state["P_ext"]
    domain_legacy["U_ext"] = state["U_ext"]
    domain_legacy["V_ext"] = state["V_ext"]
    domain_legacy["W_ext"] = state["W_ext"]
    state["Domain"] = domain_legacy

    return state
