# src/step4/initialize_staggered_fields.py

import numpy as np
from src.step4.allocate_extended_fields import allocate_extended_fields


def initialize_staggered_fields(state):
    """
    Initialize the extended staggered fields (U_ext, V_ext, W_ext, P_ext)
    using the initial conditions provided in the Step 3 configuration.

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

    # IMPORTANT:
    # allocate_extended_fields() now creates BOTH:
    #   state["Domain"]  (legacy tests)
    #   state["domain"]  (schema + pipeline)
    #
    # initialize_staggered_fields() must preserve BOTH.
    if "domain" in state:
        state["Domain"] = state["domain"]

    ic = state["config"].get("initial_conditions", {})
    p0 = ic.get("initial_pressure", 0.0)
    u0, v0, w0 = ic.get("initial_velocity", [0.0, 0.0, 0.0])

    nx = state["config"]["domain"]["nx"]
    ny = state["config"]["domain"]["ny"]
    nz = state["config"]["domain"]["nz"]

    mask = state.get("mask", None)

    # ---------------------------------------------------------
    # 2. Fill interior pressure
    # ---------------------------------------------------------
    P_ext = state["P_ext"]
    P_ext[1:nx+1, 1:ny+1, 1:nz+1] = p0

    # ---------------------------------------------------------
    # 3. Fill interior velocities (respect staggering)
    # ---------------------------------------------------------
    U_ext = state["U_ext"]
    V_ext = state["V_ext"]
    W_ext = state["W_ext"]

    # U staggered in x → shape (nx+1, ny, nz)
    U_ext[1:nx+2, 1:ny+1, 1:nz+1] = u0

    # V staggered in y → shape (nx, ny+1, nz)
    V_ext[0:nx, 1:ny+2, 1:nz+1] = v0

    # W staggered in z → shape (nx, ny, nz+1)
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

    return state
