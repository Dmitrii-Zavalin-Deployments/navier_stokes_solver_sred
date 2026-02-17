# src/step5/orchestrate_step5_state.py

from src.step3.orchestrate_step3 import orchestrate_step3_state
from src.step5.log_step_diagnostics import log_step_diagnostics
from src.step5.write_output_snapshot import write_output_snapshot
from src.step5.finalize_simulation_health import finalize_simulation_health


def orchestrate_step5_state(state):
    """
    Step 5 — Global time-integration loop.

    This function performs no physics. It repeatedly calls Step 3
    (projection method) to advance the simulation in time.

    Inputs:
        state: SolverState (must come from Step 4)
        state.config["total_time"]
        state.config["max_steps"]
        state.config["output_interval"] (optional)

    Outputs:
        state updated in place:
            - state.time
            - state.step_index
            - state.history (optional)
            - state.final_health (optional)
    """

    dt = state.constants["dt"]
    total_time = state.config["total_time"]
    max_steps = state.config.get("max_steps", 10_000)
    output_interval = state.config.get("output_interval", None)

    t = 0.0
    step = 0

    # Main time loop
    while t < total_time and step < max_steps:

        # Step 3 performs the actual physics update
        orchestrate_step3_state(state, t, step)

        # Optional: log diagnostics
        log_step_diagnostics(state, t, step)

        # Optional: write output snapshots
        if output_interval is not None and step % output_interval == 0:
            write_output_snapshot(state, t, step)

        # Advance time
        t += dt
        step += 1

    # Final bookkeeping
    state.time = t
    state.step_index = step

    # Optional summary metrics
    finalize_simulation_health(state)

    return state
