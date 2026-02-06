# src/step3/apply_boundary_conditions_post.py

def apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new):
    """
    Reapply BCs after correction.
    """

    state["U"] = U_new
    state["V"] = V_new
    state["W"] = W_new
    state["P"] = P_new

    handler = state.get("BC_handler", None)
    if handler and hasattr(handler, "apply_post"):
        handler.apply_post(state)
