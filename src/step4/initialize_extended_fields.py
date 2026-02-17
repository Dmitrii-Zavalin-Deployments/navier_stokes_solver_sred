# src/step4/initialize_extended_fields.py

import numpy as np


def initialize_extended_fields(state):
    """
    Allocate and populate extended staggered fields (P_ext, U_ext, V_ext, W_ext).

    Step‑4 responsibilities:
      • Allocate extended fields with ghost layers
      • Copy interior values from state.fields
      • Zero velocities and pressure in solid cells (state.is_fluid == False)
      • Do NOT use mask semantics (0/1/-1) — Step‑4 uses state.is_fluid only
      • Do NOT apply boundary conditions here (handled in apply_boundary_conditions)
      • Do NOT modify legacy Domain blocks or BCApplied flags
    """

    # ---------------------------------------------------------
    # Grid size
    # ---------------------------------------------------------
    nx = state.config["domain"]["nx"]
    ny = state.config["domain"]["ny"]
    nz = state.config["domain"]["nz"]

    # ---------------------------------------------------------
    # Allocate extended fields
    # ---------------------------------------------------------
    state.P_ext = np.zeros((nx + 2, ny + 2, nz + 2), dtype=float)
    state.U_ext = np.zeros((nx + 3, ny + 2, nz + 2), dtype=float)
    state.V_ext = np.zeros((nx + 2, ny + 3, nz + 2), dtype=float)
    state.W_ext = np.zeros((nx + 2, ny + 2, nz + 3), dtype=float)

    P_ext = state.P_ext
    U_ext = state.U_ext
    V_ext = state.V_ext
    W_ext = state.W_ext

    # ---------------------------------------------------------
    # Interior fields from Step‑3
    # ---------------------------------------------------------
    P = state.fields["P"]
    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]

    # Pressure interior
    P_ext[1:nx+1, 1:ny+1, 1:nz+1] = P

    # U interior (nx+1, ny, nz)
    U_ext[1:nx+2, 1:ny+1, 1:nz+1] = U

    # V interior (nx, ny+1, nz)
    V_ext[1:nx+1, 1:ny+2, 1:nz+1] = V

    # W interior (nx, ny, nz+1)
    W_ext[1:nx+1, 1:ny+1, 1:nz+2] = W

    # ---------------------------------------------------------
    # Solid zeroing using state.is_fluid
    # ---------------------------------------------------------
    if hasattr(state, "is_fluid") and state.is_fluid is not None:
        fluid = state.is_fluid
        solid = ~fluid

        # Pressure
        P_ext[1:nx+1, 1:ny+1, 1:nz+1][solid] = 0.0

        # U faces adjacent to solid cells
        U_ext[1:nx+1, 1:ny+1, 1:nz+1][solid] = 0.0
        U_ext[2:nx+2, 1:ny+1, 1:nz+1][solid] = 0.0

        # V faces adjacent to solid cells
        V_ext[1:nx+1, 1:ny+1, 1:nz+1][solid] = 0.0
        V_ext[1:nx+1, 2:ny+2, 1:nz+1][solid] = 0.0

        # W faces adjacent to solid cells
        W_ext[1:nx+1, 1:ny+1, 1:nz+1][solid] = 0.0
        W_ext[1:nx+1, 1:ny+1, 2:nz+2][solid] = 0.0

    return state
