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
    nx = state.config["grid"]["nx"]
    ny = state.config["grid"]["ny"]
    nz = state.config["grid"]["nz"]

    # ---------------------------------------------------------
    # Allocate extended fields (documentation‑accurate shapes)
    # ---------------------------------------------------------
    state.P_ext = np.zeros((nx + 2, ny + 2, nz + 2), dtype=float)
    state.U_ext = np.zeros((nx + 3, ny + 2, nz + 2), dtype=float)
    state.V_ext = np.zeros((nx,     ny + 3, nz + 2), dtype=float)
    state.W_ext = np.zeros((nx,     ny,     nz + 3), dtype=float)

    P_ext = state.P_ext
    U_ext = state.U_ext
    V_ext = state.V_ext
    W_ext = state.W_ext

    # ---------------------------------------------------------
    # Interior fields from Step‑3
    # ---------------------------------------------------------
    P = state.fields["P"]          # (nx, ny, nz)
    U = state.fields["U"]          # (nx+1, ny, nz)
    V = state.fields["V"]          # (nx, ny+1, nz)
    W = state.fields["W"]          # (nx, ny, nz+1)

    # ---------------------------------------------------------
    # Copy interior values into extended fields
    # ---------------------------------------------------------

    # Pressure interior
    P_ext[1:nx+1, 1:ny+1, 1:nz+1] = P

    # U interior (nx+1, ny, nz)
    U_ext[1:nx+2, 1:ny+1, 1:nz+1] = U

    # V interior (nx, ny+1, nz)
    # V_ext shape: (nx, ny+3, nz+2)
    V_ext[:, 1:ny+2, 1:nz+1] = V

    # W interior (nx, ny, nz+1)
    # W_ext shape: (nx, ny, nz+3)
    W_ext[:, :, 1:nz+2] = W

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
        V_ext[:, 1:ny+1, 1:nz+1][solid] = 0.0
        V_ext[:, 2:ny+2, 1:nz+1][solid] = 0.0

        # W faces adjacent to solid cells
        W_ext[:, :, 1:nz+1][solid] = 0.0
        W_ext[:, :, 2:nz+2][solid] = 0.0

    return state
