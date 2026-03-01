# src/step5/chronos_guard.py

from src.solver_state import SolverState

def synchronize_terminal_state(state: SolverState) -> None:
    """
    Point 2: Terminal Temporal State.
    Ensures state.time does not overshoot total_time.
    """
    total_time = state.config.total_time
    
    if state.time >= total_time:
        state.time = total_time
        # Progression Gate: Signal the MainSolver to stop
        state.ready_for_time_loop = False
    
    # Update health vitals for the final summary
    state.health.is_stable = True
    state.health.post_correction_divergence_norm = state.health.divergence_norm