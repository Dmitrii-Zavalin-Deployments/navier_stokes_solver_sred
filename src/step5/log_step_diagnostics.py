# src/step5/log_step_diagnostics.py

def log_step_diagnostics(state, current_time, step_index):
    """
    Append timestep diagnostics to state.history.

    Step 3 updates:
        - state.health["post_correction_divergence_norm"]
        - state.health["max_velocity_magnitude"]
        - state.health["cfl_advection_estimate"]
        - state.ppe["iterations"]
    """

    # Initialize history if needed
    if state.history is None:
        state.history = {
            "times": [],
            "steps": [],
            "divergence_norms": [],
            "max_velocity_history": [],
            "cfl_values": [],
            "ppe_iterations": [],
        }

    state.history["times"].append(current_time)
    state.history["steps"].append(step_index)
    state.history["divergence_norms"].append(
        state.health.get("post_correction_divergence_norm", 0.0)
    )
    state.history["max_velocity_history"].append(
        state.health.get("max_velocity_magnitude", 0.0)
    )
    state.history["cfl_values"].append(
        state.health.get("cfl_advection_estimate", 0.0)
    )
    state.history["ppe_iterations"].append(
        state.ppe.get("iterations", 0)
    )
