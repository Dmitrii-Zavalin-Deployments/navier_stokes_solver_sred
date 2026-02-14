# file: src/step4/allocate_extended_fields.py

import numpy as np


def allocate_extended_fields(state):
    cfg = state["config"]["domain"]
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]

    fields = state["fields"]

    # ---------------------------------------------------------
    # Allocate extended arrays (axis order: x, y, z)
    # ---------------------------------------------------------
    P_ext = np.zeros((nx + 2, ny + 2, nz + 2), dtype=float)
    U_ext = np.zeros((nx + 3, ny + 2, nz + 2), dtype=float)
    V_ext = np.zeros((nx,     ny + 3, nz + 2), dtype=float)
    W_ext = np.zeros((nx,     ny,     nz + 3), dtype=float)

    # ---------------------------------------------------------
    # Copy interior values
    # ---------------------------------------------------------
    if "P" in fields:
        P_ext[1:nx+1, 1:ny+1, 1:nz+1] = fields["P"]

    if "U" in fields:
        # Correct alignment so that:
        # out["U_ext"][1:-2, 1:-1, 1:-1] == U
        #
        # U_ext.shape[0] = nx+3
        # slice 1:-2 → indices 1..nx+1 → length nx+1
        #
        # So we must copy U into indices 1..nx+1 inclusive.
        U_ext[1:nx+2, 1:ny+1, 1:nz+1] = fields["U"]

    if "V" in fields:
        V_ext[:, 1:ny+2, 1:nz+1] = fields["V"]

    if "W" in fields:
        W_ext[:, :, 1:nz+2] = fields["W"]

    # ---------------------------------------------------------
    # Store extended fields
    # ---------------------------------------------------------
    state["P_ext"] = P_ext
    state["U_ext"] = U_ext
    state["V_ext"] = V_ext
    state["W_ext"] = W_ext

    # ---------------------------------------------------------
    # GhostLayers must return INDEX TUPLES, not array slices
    # ---------------------------------------------------------
    def ghost_indices(arr):
        return {
            "GHOST_X_LO": (0, slice(None), slice(None)),
            "GHOST_X_HI": (-1, slice(None), slice(None)),
            "GHOST_Y_LO": (slice(None), 0, slice(None)),
            "GHOST_Y_HI": (slice(None), -1, slice(None)),
            "GHOST_Z_LO": (slice(None), slice(None), 0),
            "GHOST_Z_HI": (slice(None), slice(None), -1),
        }

    GhostLayers = {
        "P_ext": ghost_indices(P_ext),
        "U_ext": ghost_indices(U_ext),
        "V_ext": ghost_indices(V_ext),
        "W_ext": ghost_indices(W_ext),
    }

    # ---------------------------------------------------------
    # Domain block
    # ---------------------------------------------------------
    state["Domain"] = {
        "P_ext": P_ext,
        "U_ext": U_ext,
        "V_ext": V_ext,
        "W_ext": W_ext,

        "GhostLayers": GhostLayers,

        "index_ranges": {
            "x": (0, nx - 1),
            "y": (0, ny - 1),
            "z": (0, nz - 1),
        },

        "views": {
            "P_interior": P_ext[1:nx+1, 1:ny+1, 1:nz+1],
            "U_interior": U_ext[1:nx+2, 1:ny+1, 1:nz+1],
            "V_interior": V_ext[:, 1:ny+2, 1:nz+1],
            "W_interior": W_ext[:, :, 1:nz+2],
        },
    }

    return state
