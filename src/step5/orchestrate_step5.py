# src/step5/orchestrate_step5_state.py

from src.step3.orchestrate_step3 import orchestrate_step3
from src.step5.log_step_diagnostics import log_step_diagnostics
from src.step5.write_output_snapshot import write_output_snapshot
from src.step5.finalize_simulation_health import finalize_simulation_health

def orchestrate_step5(state):
    """
    Step 5 — Global time-integration loop.
    Standardized to call orchestrate_step3.
    """
    dt = state.constants["dt"]
    total_time = state.config["total_time"]
    max_steps = state.config.get("max_steps", 10_000)
    output_interval = state.config.get("output_interval", None)

    t = 0.0
    step = 0

    # Main time loop
    while t < total_time and step < max_steps:

        # Fixed: now uses the standardized name
        orchestrate_step3(state, t, step)

        log_step_diagnostics(state, t, step)

        if output_interval is not None and step % output_interval == 0:
            write_output_snapshot(state, t, step)

        t += dt
        step += 1

    state.time = t
    state.step_index = step
    finalize_simulation_health(state)

    return state