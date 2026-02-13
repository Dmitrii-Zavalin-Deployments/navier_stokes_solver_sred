# src/step4/assemble_history.py

def assemble_history(state):
    """
    Build the schema-compliant history block for Step 4 output.

    Step 4 does not perform time-stepping, so all history arrays
    must be initialized as empty lists.
    """

    state["history"] = {
        "times": [],
        "divergence_norms": [],
        "max_velocity_history": [],
        "ppe_iterations_history": [],
    }

    return state
