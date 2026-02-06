# src/step3/log_step_diagnostics.py

import numpy as np

def log_step_diagnostics(state, current_time, step_index):
    """
    Append diagnostics to History.
    """

    hist = state.setdefault("History", {})
    for key in ["times", "divergence_norms", "max_velocity_history",
                "ppe_iterations_history", "energy_history"]:
        hist.setdefault(key, [])

    hist["times"].append(float(current_time))
    hist["divergence_norms"].append(state["Health"]["post_correction_divergence_norm"])
    hist["max_velocity_history"].append(state["Health"]["max_velocity_magnitude"])
    hist["ppe_iterations_history"].append(state["PPE"].get("last_iterations", -1))

    U, V, W = state["U"], state["V"], state["W"]
    rho = state["Constants"]["rho"]

    energy = 0.5 * rho * (np.mean(U**2) + np.mean(V**2) + np.mean(W**2))
    hist["energy_history"].append(float(energy))
