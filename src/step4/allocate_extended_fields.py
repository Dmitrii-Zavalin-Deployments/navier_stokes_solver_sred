# file: src/step4/allocate_extended_fields.py

import numpy as np


def allocate_extended_fields(state):
    """
    Allocate extended (halo) fields for Step 4.

    Produces uppercase *_ext fields that match the Step‑4 schema:

        P_ext: (nz+2, ny+2, nx+2)
        U_ext: (nz+2, ny+2, nx+3)
        V_ext: (nz+2, ny+3, nx+2)
        W_ext: (nz+3, ny+2, nx+2)

    The interior values from P, U, V, W are copied into the
    [1:-1, 1:-1, 1:-1] region of each extended array.

    The original fields remain unchanged.
    """

    # ---------------------------------------------------------
    # Grid sizes
    # ---------------------------------------------------------
    config_domain = state.get("config", {}).get("domain", {})
    nx = config_domain.get("nx", 1)
    ny = config_domain.get("ny", 1)
    nz = config_domain.get("nz", 1)

    # ---------------------------------------------------------
    # Allocate extended arrays with correct shapes
    # ---------------------------------------------------------
    P_ext = np.zeros((nz + 2, ny + 2, nx + 2))
    U_ext = np.zeros((nz + 2, ny + 2, nx + 3))
    V_ext = np.zeros((nz + 2, ny + 3, nx + 2))
    W_ext = np.zeros((nz + 3, ny + 2, nx + 2))

    # ---------------------------------------------------------
    # Copy interior values (if present)
    # ---------------------------------------------------------
    if "P" in state and isinstance(state["P"], np.ndarray):
        P_ext[1:-1, 1:-1, 1:-1] = state["P"]

    if "U" in state and isinstance(state["U"], np.ndarray):
        # U is staggered in x → interior shape (nz, ny, nx+1)
        U_ext[1:-1, 1:-1, 1:-2] = state["U"]

    if "V" in state and isinstance(state["V"], np.ndarray):
        # V is staggered in y → interior shape (nz, ny+1, nx)
        V_ext[1:-1, 1:-2, 1:-1] = state["V"]

    if "W" in state and isinstance(state["W"], np.ndarray):
        # W is staggered in z → interior shape (nz+1, ny, nx)
        W_ext[1:-2, 1:-1, 1:-1] = state["W"]

    # ---------------------------------------------------------
    # Store uppercase extended fields (schema truth)
    # ---------------------------------------------------------
    state["P_ext"] = P_ext
    state["U_ext"] = U_ext
    state["V_ext"] = V_ext
    state["W_ext"] = W_ext

    return state
