# src/step5/finalize_simulation_health.py

def finalize_simulation_health(state):
    """
    Compute summary metrics after the time loop completes.
    """

    history = state.history or {}

    ppe_iters = history.get("ppe_iterations", [])
    cfl_vals = history.get("cfl_values", [])

    avg_ppe = sum(ppe_iters) / len(ppe_iters) if ppe_iters else 0.0
    max_cfl = max(cfl_vals) if cfl_vals else 0.0

    state.final_health = {
        "final_time": state.time,
        "total_steps_taken": state.step_index,
        "final_divergence_norm": state.health.get("post_correction_divergence_norm", 0.0),
        "final_max_velocity": state.health.get("max_velocity_magnitude", 0.0),
        "average_ppe_iterations": avg_ppe,
        "max_cfl_encountered": max_cfl,
        "simulation_success": True,
        "termination_reason": "normal",
    }
