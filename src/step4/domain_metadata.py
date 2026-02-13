# file: src/step4/domain_metadata.py

import numpy as np


def build_domain_block(state):
    """
    Build a schema-compliant 'domain' block for Step 4 output.

    Uses:
    - config.domain.{nx,ny,nz}
    - extended fields (P_ext/U_ext/V_ext/W_ext)
    to infer sizes and ghost layer structure.
    """

    # ---------------------------------------------------------
    # Extract grid sizes
    # ---------------------------------------------------------
    config_domain = state.get("config", {}).get("domain", {})
    nx = config_domain.get("nx", 1)
    ny = config_domain.get("ny", 1)
    nz = config_domain.get("nz", 1)

    # ---------------------------------------------------------
    # Access extended fields (uppercase only — schema truth)
    # ---------------------------------------------------------
    P_ext = state.get("P_ext")
    U_ext = state.get("U_ext")
    V_ext = state.get("V_ext")
    W_ext = state.get("W_ext")

    # ---------------------------------------------------------
    # Infer ghost layers from NumPy array shapes
    # ---------------------------------------------------------
    def infer_ghost_layers(ext_arr, interior_shape):
        """
        ext_arr: NumPy array with ghost layers
        interior_shape: (nx, ny, nz) for the corresponding field

        Returns [lo, hi] ghost counts along the axis of interest.
        """
        if ext_arr is None:
            return [0, 0]

        ext_n = ext_arr.shape
        int_n = interior_shape

        # Ghost layers = (extended - interior)
        ghost_total = max(0, ext_n - int_n)

        lo = ghost_total // 2
        hi = ghost_total - lo
        return [int(lo), int(hi)]

    # Pressure field is cell-centered → shape (nx, ny, nz)
    ghost_Px = infer_ghost_layers(P_ext.shape[2], nx)
    ghost_Py = infer_ghost_layers(P_ext.shape[1], ny)
    ghost_Pz = infer_ghost_layers(P_ext.shape[0], nz)

    # U is staggered in x → interior shape (nx+1, ny, nz)
    ghost_Ux = infer_ghost_layers(U_ext.shape[2], nx + 1)
    ghost_Uy = infer_ghost_layers(U_ext.shape[1], ny)
    ghost_Uz = infer_ghost_layers(U_ext.shape[0], nz)

    # V is staggered in y → interior shape (nx, ny+1, nz)
    ghost_Vx = infer_ghost_layers(V_ext.shape[2], nx)
    ghost_Vy = infer_ghost_layers(V_ext.shape[1], ny + 1)
    ghost_Vz = infer_ghost_layers(V_ext.shape[0], nz)

    # W is staggered in z → interior shape (nx, ny, nz+1)
    ghost_Wx = infer_ghost_layers(W_ext.shape[2], nx)
    ghost_Wy = infer_ghost_layers(W_ext.shape[1], ny)
    ghost_Wz = infer_ghost_layers(W_ext.shape[0], nz + 1)

    ghost_layers = {
        "P_ext": [ghost_Px, ghost_Py, ghost_Pz],
        "U_ext": [ghost_Ux, ghost_Uy, ghost_Uz],
        "V_ext": [ghost_Vx, ghost_Vy, ghost_Vz],
        "W_ext": [ghost_Wx, ghost_Wy, ghost_Wz],
    }

    # ---------------------------------------------------------
    # Coordinates (uniform grid for now)
    # ---------------------------------------------------------
    def linspace_1d(n):
        if n <= 0:
            return []
        if n == 1:
            return [0.0]
        return [float(i) / float(n - 1) for i in range(n)]

    coordinates = {
        "x_centers": linspace_1d(nx),
        "y_centers": linspace_1d(ny),
        "z_centers": linspace_1d(nz),
        "x_faces_u": linspace_1d(nx + 1),
        "y_faces_v": linspace_1d(ny + 1),
        "z_faces_w": linspace_1d(nz + 1),
    }

    # ---------------------------------------------------------
    # Index ranges (string‑encoded)
    # ---------------------------------------------------------
    index_ranges = {
        "interior": f"i=0..{nx-1},j=0..{ny-1},k=0..{nz-1}",
        "ghost_x_lo": "i<0",
        "ghost_x_hi": f"i>={nx}",
        "ghost_y_lo": "j<0",
        "ghost_y_hi": f"j>={ny}",
        "ghost_z_lo": "k<0",
        "ghost_z_hi": f"k>={nz}",
    }

    # ---------------------------------------------------------
    # Stencil maps (simple but valid)
    # ---------------------------------------------------------
    stencil_maps = {
        "xp": [1],
        "xm": [-1],
        "yp": [1],
        "ym": [-1],
        "zp": [1],
        "zm": [-1],
    }

    # ---------------------------------------------------------
    # Interpolation maps (empty but schema‑valid)
    # ---------------------------------------------------------
    interpolation_maps = {
        "interp_u_to_v": {},
        "interp_u_to_w": {},
        "interp_v_to_u": {},
        "interp_v_to_w": {},
        "interp_w_to_u": {},
        "interp_w_to_v": {},
    }

    # ---------------------------------------------------------
    # Assemble final domain block
    # ---------------------------------------------------------
    state["domain"] = {
        "coordinates": coordinates,
        "ghost_layers": ghost_layers,
        "index_ranges": index_ranges,
        "stencil_maps": stencil_maps,
        "interpolation_maps": interpolation_maps,
    }

    return state
