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
    # Copy interior values (defensively)
    # ---------------------------------------------------------

    # ---------- P ----------
    if "P" in fields:
        P = np.asarray(fields["P"])
        sx = min(nx, P.shape[0])
        sy = min(ny, P.shape[1])
        sz = min(nz, P.shape[2])
        P_ext[1:1+sx, 1:1+sy, 1:1+sz] = P[:sx, :sy, :sz]

    # ---------- U (staggered in x) ----------
    if "U" in fields:
        U = np.asarray(fields["U"])
        sx = min(nx + 1, U.shape[0])
        sy = min(ny,     U.shape[1])
        sz = min(nz,     U.shape[2])
        U_ext[1:1+sx, 1:1+sy, 1:1+sz] = U[:sx, :sy, :sz]

    # ---------- V (staggered in y) ----------
    if "V" in fields:
        V = np.asarray(fields["V"])
        sx = min(nx,     V.shape[0])
        sy = min(ny + 1, V.shape[1])
        sz = min(nz,     V.shape[2])
        V_ext[0:sx, 1:1+sy, 1:1+sz] = V[:sx, :sy, :sz]

    # ---------- W (staggered in z) ----------
    if "W" in fields:
        W = np.asarray(fields["W"])
        sx = min(nx,     W.shape[0])
        sy = min(ny,     W.shape[1])
        sz = min(nz + 1, W.shape[2])
        W_ext[0:sx, 0:sy, 1:1+sz] = W[:sx, :sy, :sz]

    # ---------------------------------------------------------
    # Store extended fields at top level (schema requirement)
    # ---------------------------------------------------------
    state["P_ext"] = P_ext
    state["U_ext"] = U_ext
    state["V_ext"] = V_ext
    state["W_ext"] = W_ext

    # ---------------------------------------------------------
    # Schema-compliant ghost layer format: [lo, hi]
    # ---------------------------------------------------------
    ghost_layers = {
        "P_ext": [1, 1],
        "U_ext": [1, 1],
        "V_ext": [1, 1],
        "W_ext": [1, 1],
    }

    # ---------------------------------------------------------
    # Domain block (lowercase — required by schema)
    # ---------------------------------------------------------
    state["domain"] = {
        "ghost_layers": ghost_layers,

        # Filled later by build_domain_block()
        "coordinates": {},
        "index_ranges": {},
        "stencil_maps": {},
        "interpolation_maps": {},

        # Views and index ranges required by allocate_extended_fields tests
        "views": {
            "P_interior": P_ext[1:nx+1, 1:ny+1, 1:nz+1],
            "U_interior": U_ext[1:nx+2, 1:ny+1, 1:nz+1],
            "V_interior": V_ext[:, 1:ny+2, 1:nz+1],
            "W_interior": W_ext[:, :, 1:nz+2],
        },

        "index_ranges_internal": {
            "x": (0, nx - 1),
            "y": (0, ny - 1),
            "z": (0, nz - 1),
        },
    }

    return state
