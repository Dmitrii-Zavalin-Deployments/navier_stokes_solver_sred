# src/step4/initialize_staggered_fields.py

import numpy as np
from src.step4.allocate_extended_fields import allocate_extended_fields


def initialize_staggered_fields(state):
    """
    Initialize the extended staggered fields (U_ext, V_ext, W_ext, P_ext)
    using the initial conditions provided in the configuration.

    Responsibilities:
    - Allocate extended fields (via allocate_extended_fields).
    - Fill interior pressure and velocity with initial conditions.
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

    # Mask: ensure proper 3D NumPy shape (nx, ny, nz) if present
    mask_raw = state.get("mask", None)
    if mask_raw is not None:
        mask = np.asarray(mask_raw, dtype=int).reshape((nx, ny, nz))
    else:
        mask = None

    P_ext = state["P_ext"]
    U_ext = state["U_ext"]
    V_ext = state["V_ext"]
    W_ext = state["W_ext"]

    # ---------------------------------------------------------
    # 2. Fill interior pressure with initial condition
    # ---------------------------------------------------------
    P_ext[1:nx+1, 1:ny+1, 1:nz+1] = p0

    # ---------------------------------------------------------
    # 3. Fill interior velocities (respect staggering)
    # ---------------------------------------------------------
    # U staggered in x → shape (nx+3, ny+2, nz+2), interior (nx+1, ny, nz)
    U_ext[1:nx+2, 1:ny+1, 1:nz+1] = u0

    # V staggered in y → shape (nx, ny+3, nz+2), interior (nx, ny+1, nz)
    V_ext[0:nx, 1:ny+2, 1:nz+1] = v0

    # W staggered in z → shape (nx, ny, nz+3), interior (nx, ny, nz+1)
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
        P_ext[1:nx+1, 1:ny+1, 1:nz+1][boundary_fluid] = p0

        # U faces
        U_ext[1:nx+1, 1:ny+1, 1:nz+1][boundary_fluid] = u0
        U_ext[2:nx+2, 1:ny+1, 1:nz+1][boundary_fluid] = u0

        # V faces
        V_ext[0:nx, 1:ny+1, 1:nz+1][boundary_fluid] = v0
        V_ext[0:nx, 2:ny+2, 1:nz+1][boundary_fluid] = v0

        # W faces
        W_ext[0:nx, 0:ny, 1:nz+1][boundary_fluid] = w0
        W_ext[0:nx, 0:ny, 2:nz+2][boundary_fluid] = w0

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
