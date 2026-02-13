# file: src/step4/domain_metadata.py

def build_domain_block(state):
    """
    Build a schema-compliant 'domain' block for Step 4 output.

    Uses:
    - config.domain.{nx,ny,nz}
    - extended fields (P_ext/U_ext/V_ext/W_ext)
    to define coordinates and simple ghost-layer metadata.
    """

    # ---------------------------------------------------------
    # Extract grid sizes
    # ---------------------------------------------------------
    config_domain = state.get("config", {}).get("domain", {})
    nx = config_domain.get("nx", 1)
    ny = config_domain.get("ny", 1)
    nz = config_domain.get("nz", 1)

    # ---------------------------------------------------------
    # Ghost layers (simple, schema-valid placeholders)
    # ---------------------------------------------------------
    # Schema only requires "array" type, not specific length/values.
    # We use a simple 6-entry convention [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi].
    ghost_layers = {
        "P_ext": [1, 1, 1, 1, 1, 1],
        "U_ext": [1, 1, 1, 1, 1, 1],
        "V_ext": [1, 1, 1, 1, 1, 1],
        "W_ext": [1, 1, 1, 1, 1, 1],
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
