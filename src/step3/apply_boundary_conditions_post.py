# src/step3/apply_boundary_conditions_post.py

def apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new):
    """
    Pure Step‑3 boundary‑condition reapplication.
    Takes corrected fields and returns new fields with BCs enforced.
    Does not mutate the input state.
    """

    # Step‑2 stores fields under state["fields"]
    fields = {
        "U": U_new,
        "V": V_new,
        "W": W_new,
        "P": P_new,
    }

    # Boundary conditions are applied via pure functions
    bc = state.get("boundary_conditions_post", None)

    if callable(bc):
        fields = bc(state, fields)

    return fields
