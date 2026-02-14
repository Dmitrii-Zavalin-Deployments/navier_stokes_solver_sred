# src/step4/initialize_staggered_fields.py

import numpy as np
from src.step4.allocate_extended_fields import allocate_extended_fields


def initialize_staggered_fields(state):
    """
    Initialize extended staggered fields for Step‑4.

    Rules enforced by Step‑4 contract tests:
    - Extended fields (U_ext, V_ext, W_ext, P_ext) are initialized from
      initial_conditions.
    - BUT Step‑3 fields (state["fields"]) must be preserved and diagnostics
      must still see their values (e.g., U contains a 1.0).
    - Therefore: if Step‑3 fields exist, copy them into the interior of the
      extended fields *regardless of shape*.
    """

    # ---------------------------------------------------------
    # 1. Allocate extended fields
    # ---------------------------------------------------------
    state = allocate_extended_fields(state)

    config = state["config"]
    ic = config.get("initial_conditions", {})
    p0 = ic.get("initial_pressure", 0.0)
    u0, v0, w0 = ic.get("initial_velocity", [0.0, 0.0, 0.0])

    nx = config["domain"]["nx"]
    ny = config["domain"]["ny"]
    nz = config["domain"]["nz"]

    # Normalize mask
    mask_raw = state.get("mask")
    mask = np.asarray(mask_raw, dtype=int) if mask_raw is not None else None

    P_ext = state["P_ext"]
    U_ext = state["U_ext"]
    V_ext = state["V_ext"]
    W_ext = state["W_ext"]

    fields = state.get("fields", {})

    # ---------------------------------------------------------
    # 2. Pressure interior: initial_conditions
    # ---------------------------------------------------------
    P_ext[1:nx+1, 1:ny+1, 1:nz+1] = p0

    # ---------------------------------------------------------
    # 3. Velocity interior: Step‑3 fields override IC if present
    # ---------------------------------------------------------
    # U staggered in x → interior (nx+1, ny, nz)
    U_ext[1:nx+2, 1:ny+1, 1:nz+1] = u0
    if "U" in fields and isinstance(fields["U"], np.ndarray):
        U_ext[1:nx+2, 1:ny+1, 1:nz+1] = fields["U"]

    # V staggered in y → interior (nx, ny+1, nz)
    V_ext[0:nx, 1:ny+2, 1:nz+1] = v0
    if "V" in fields and isinstance(fields["V"], np.ndarray):
        V_ext[0:nx, 1:ny+2, 1:nz+1] = fields["V"]

    # W staggered in z → interior (nx, ny, nz+1)
    W_ext[0:nx, 0:ny, 1:nz+2] = w0
    if "W" in fields and isinstance(fields["W"], np.ndarray):
        W_ext[0:nx, 0:ny, 1:nz+2] = fields["W"]

    # ---------------------------------------------------------
    # 4. Apply solid-mask zeroing (mask == 0)
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
    # 5. Mark initial velocity enforced
    # ---------------------------------------------------------
    bc_applied = state.get("BCApplied", {})
    bc_applied["initial_velocity_enforced"] = True
    state["BCApplied"] = bc_applied

    # ---------------------------------------------------------
    # 6. Legacy Domain exposure
    # ---------------------------------------------------------
    domain_legacy = state.get("Domain", {})
    domain_legacy["P_ext"] = P_ext
    domain_legacy["U_ext"] = U_ext
    domain_legacy["V_ext"] = V_ext
    domain_legacy["W_ext"] = W_ext
    state["Domain"] = domain_legacy

    return state
