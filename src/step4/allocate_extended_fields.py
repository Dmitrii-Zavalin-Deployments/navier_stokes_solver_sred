import numpy as np


def allocate_extended_fields(state):
    """
    Allocate extended (ghost-layer) fields for Step 4.

    Axis order is ALWAYS (x, y, z).

    Shapes required by tests:
        P_ext: (nx+2, ny+2, nz+2)
        U_ext: (nx+3, ny+2, nz+2)
        V_ext: (nx+2, ny+3, nz+2)
        W_ext: (nx+2, ny+2, nz+3)

    Interior values are copied into [1:-1, 1:-1, 1:-1].
    Ghost layers are zero-filled.
    """

    cfg = state["config"]["domain"]
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]

    fields = state["fields"]

    # ---------------------------------------------------------
    # Allocate extended arrays (axis order: x, y, z)
    # ---------------------------------------------------------
    P_ext = np.zeros((nx + 2, ny + 2, nz + 2), dtype=float)
    U_ext = np.zeros((nx + 3, ny + 2, nz + 2), dtype=float)
    V_ext = np.zeros((nx + 2, ny + 3, nz + 2), dtype=float)
    W_ext = np.zeros((nx + 2, ny + 2, nz + 3), dtype=float)

    # ---------------------------------------------------------
    # Copy interior values
    # ---------------------------------------------------------
    if "P" in fields:
        P_ext[1:nx+1, 1:ny+1, 1:nz+1] = fields["P"]

    if "U" in fields:
        # U interior shape: (nx+1, ny, nz)
        U_ext[1:nx+2, 1:ny+1, 1:nz+1] = fields["U"]

    if "V" in fields:
        # V interior shape: (nx, ny+1, nz)
        V_ext[1:nx+1, 1:ny+2, 1:nz+1] = fields["V"]

    if "W" in fields:
        # W interior shape: (nx, ny, nz+1)
        W_ext[1:nx+1, 1:ny+1, 1:nz+2] = fields["W"]

    # ---------------------------------------------------------
    # Store extended fields
    # ---------------------------------------------------------
    state["P_ext"] = P_ext
    state["U_ext"] = U_ext
    state["V_ext"] = V_ext
    state["W_ext"] = W_ext

    # ---------------------------------------------------------
    # Insert "Domain" block required by tests
    # ---------------------------------------------------------
    state["Domain"] = {
        "P_ext": P_ext,
        "U_ext": U_ext,
        "V_ext": V_ext,
        "W_ext": W_ext,
        "index_ranges": {
            "x": (0, nx - 1),
            "y": (0, ny - 1),
            "z": (0, nz - 1),
        },
        "views": {
            "P_interior": P_ext[1:nx+1, 1:ny+1, 1:nz+1],
            "U_interior": U_ext[1:nx+2, 1:ny+1, 1:nz+1],
            "V_interior": V_ext[1:nx+1, 1:ny+2, 1:nz+1],
            "W_interior": W_ext[1:nx+1, 1:ny+1, 1:nz+2],
        },
    }

    return state
