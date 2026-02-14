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
    - Preserve boundary-fluid cells (mask == -1) by never zeroing them.
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

    # ---------------------------------------------------------
    # 2. Normalize mask to a NumPy array with shape (nx, ny, nz) if present
    # ---------------------------------------------------------
    mask_raw = state.get("mask", None)
    mask = None
    if mask_raw is not None:
        mask_arr = np.asarray(mask_raw, dtype=int)
        if mask_arr.shape != (nx, ny, nz):
            # Broadcast to (nx, ny, nz) if possible
            try:
                mask = np.broadcast_to(mask_arr, (nx, ny, nz))
            except ValueError as exc:
                raise RuntimeError(
                    f"[Step 4] Cannot broadcast mask of shape {mask_arr.shape} "
                    f"to domain shape ({nx}, {ny}, {nz})"
                ) from exc
        else:
            mask = mask_arr

    P_ext = state["P_ext"]
    U_ext = state["U_ext"]
    V_ext = state["V_ext"]
    W_ext = state["W_ext"]

    # ---------------------------------------------------------
    # 3. Fill interior pressure with initial_conditions
    # ---------------------------------------------------------
    P_ext[1:nx+1, 1:ny+1, 1:nz+1] = p0

    # ---------------------------------------------------------
    # 4. Fill interior velocities (respect staggering) with initial_conditions
    # ---------------------------------------------------------
    # U staggered in x → interior shape (nx+1, ny, nz)
    U_ext[1:nx+2, 1:ny+1, 1:nz+1] = u0

    # V staggered in y → interior shape (nx, ny+1, nz)
    V_ext[0:nx, 1:ny+2, 1:nz+1] = v0

    # W staggered in z → interior shape (nx, ny, nz+1)
    W_ext[0:nx, 0:ny, 1:nz+2] = w0

    # ---------------------------------------------------------
    # 5. Apply solid-mask zeroing (mask == 0)
    #    Boundary-fluid cells (mask == -1) are never zeroed.
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
    domain_legacy["P_ext"] = state["P_ext"]
    domain_legacy["U_ext"] = state["U_ext"]
    domain_legacy["V_ext"] = state["V_ext"]
    domain_legacy["W_ext"] = state["W_ext"]
    state["Domain"] = domain_legacy

    return state
