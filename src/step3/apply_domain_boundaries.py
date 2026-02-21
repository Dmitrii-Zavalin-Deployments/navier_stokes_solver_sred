# src/step3/apply_domain_boundaries.py

import numpy as np

def apply_domain_boundaries(state, fields):
    """
    Surgically applies domain-level boundary conditions (x_min, x_max, etc.)
    based on the JSON config enum: no-slip, free-slip, inflow, outflow.
    """
    # Work on copies to ensure we don't mutate the input dictionary unexpectedly
    U = fields["U"]
    V = fields["V"]
    W = fields["W"]
    P = fields["P"]
    
    bc_list = state.config.get("boundary_conditions", [])
    
    for bc in bc_list:
        loc = bc["location"]
        b_type = bc["type"]
        vals = bc.get("values", {})
        
        # Component mapping: loc string -> (axis, index, velocity_component)
        mapping = {
            "x_min": (0, 0, 'u'), "x_max": (0, -1, 'u'),
            "y_min": (1, 0, 'v'), "y_max": (1, -1, 'v'),
            "z_min": (2, 0, 'w'), "z_max": (2, -1, 'w')
        }
        
        if loc in mapping:
            axis, idx, comp = mapping[loc]
            _enforce_face(U, V, W, axis=axis, index=idx, b_type=b_type, vals=vals, component=comp)

    return {"U": U, "V": V, "W": W, "P": P}

def _enforce_face(U, V, W, axis, index, b_type, vals, component):
    target_array = {"u": U, "v": V, "w": W}[component]
    
    slc = [slice(None)] * 3
    slc[axis] = index
    slc = tuple(slc)

    if b_type in ["no-slip", "free-slip"]:
        target_array[slc] = 0.0
    
    elif b_type == "inflow":
        v_val = vals.get(component, 0.0)
        target_array[slc] = v_val
        
    elif b_type == "outflow":
        # Neighbor Rule for zero-gradient
        neighbor_idx = 1 if index == 0 else -2
        slc_neigh = [slice(None)] * 3
        slc_neigh[axis] = neighbor_idx
        # Use np.copy to prevent potential view/reference issues during the update
        target_array[slc] = np.copy(target_array[tuple(slc_neigh)])