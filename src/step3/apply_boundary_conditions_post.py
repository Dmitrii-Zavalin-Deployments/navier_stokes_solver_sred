# src/step3/apply_boundary_conditions_post.py

def apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new):
    """
    Step‑3 boundary‑condition reapplication.
    Takes corrected fields and returns new fields with BCs enforced.
    Does not mutate the input state.
    """

    # Pack corrected fields into a temporary structure
    fields = {
        "U": U_new,
        "V": V_new,
        "W": W_new,
        "P": P_new,
    }

    # Boundary conditions are stored in state.boundary_conditions
    bc_handler = state.boundary_conditions

    # If a callable BC handler exists, apply it
    if callable(bc_handler):
        fields = bc_handler(state, fields)

    return fields
