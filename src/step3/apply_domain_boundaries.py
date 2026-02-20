# src/step3/apply_domain_boundaries.py

import numpy as np

def apply_domain_boundaries(state, fields):
    """
    Surgically applies domain-level boundary conditions (x_min, x_max, etc.)
    based on the JSON config enum: no-slip, free-slip, inflow.
    
    Direct Indexing Architecture: Uses slices to avoid ghost cells and minimize memory.
    """
    U = fields["U"]
    V = fields["V"]
    W = fields["W"]
    
    # boundary_conditions in config is a list of objects per the schema
    bc_list = state.config.get("boundary_conditions", [])
    
    for bc in bc_list:
        loc = bc["location"] # e.g., "x_min"
        b_type = bc["type"]  # e.g., "no-slip"
        vals = bc.get("values", {})
        
        # --- X Boundaries (Affect U normal component) ---
        if loc == "x_min":
            _enforce_face(U, V, W, axis=0, index=0, b_type=b_type, vals=vals, component='u')
        elif loc == "x_max":
            _enforce_face(U, V, W, axis=0, index=-1, b_type=b_type, vals=vals, component='u')
            
        # --- Y Boundaries (Affect V normal component) ---
        elif loc == "y_min":
            _enforce_face(U, V, W, axis=1, index=0, b_type=b_type, vals=vals, component='v')
        elif loc == "y_max":
            _enforce_face(U, V, W, axis=1, index=-1, b_type=b_type, vals=vals, component='v')
            
        # --- Z Boundaries (Affect W normal component) ---
        elif loc == "z_min":
            _enforce_face(U, V, W, axis=2, index=0, b_type=b_type, vals=vals, component='w')
        elif loc == "z_max":
            _enforce_face(U, V, W, axis=2, index=-1, b_type=b_type, vals=vals, component='w')

    return {"U": U, "V": V, "W": W, "P": fields["P"]}

def _enforce_face(U, V, W, axis, index, b_type, vals, component):
    """Helper to apply specific enum logic to a face slice."""
    # Target the normal velocity array for this face
    target_array = {"u": U, "v": V, "w": W}[component]
    
    # Slice selection: e.g., if axis=0, index=0 -> [0, :, :]
    slc = [slice(None)] * 3
    slc[axis] = index
    slc = tuple(slc)

    if b_type == "no-slip":
        # Normal component is 0. 
        # (Tangential components would be handled via ghost cells or zeroed 
        # in the internal mask logic if touching a solid).
        target_array[slc] = 0.0
    
    elif b_type == "free-slip":
        # Normal component is 0 (No-penetration)
        target_array[slc] = 0.0
        
    elif b_type == "inflow":
        # Set normal component to specified value
        v_val = vals.get(component, 0.0)
        target_array[slc] = v_val
        
    elif b_type == "outflow":
        # Zero-gradient (Neumann) for the normal component
        # U[index] = U[neighbor]
        neighbor_idx = 1 if index == 0 else -2
        slc_neigh = [slice(None)] * 3
        slc_neigh[axis] = neighbor_idx
        target_array[slc] = target_array[tuple(slc_neigh)]