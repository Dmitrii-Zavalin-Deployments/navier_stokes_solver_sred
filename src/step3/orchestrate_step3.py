# src/step3/orchestrate_step3.py

from src.solver_state import SolverState
from .predictor import predict_velocity
from .solver import solve_pressure
from .corrector import correct_velocity

# Global Debug Toggle
DEBUG = True

def orchestrate_step3(state: SolverState) -> SolverState:
    """
    Step 3 Orchestrator: Physical Integration.
    Sequence: Predict (V*) -> Solve (P) -> Correct (V_new).
    
    Rule 5 Compliance: No silent failures. No defaults for history.
    """
    if DEBUG:
        print(f"\nDEBUG [Orchestrator]: Starting Step 3 (Time: {state.time})")

    # 1. PREDICT: Calculate intermediate V*
    predict_velocity(state)
    
    # 2. SOLVE: Solve the Pressure Poisson Equation (PPE)
    status = solve_pressure(state)
    
    # CRITICAL CHECK: If the solver failed/diverged, do not apply correction.
    if status != "converged":
        if DEBUG:
            print(f"!!! CRITICAL: Pressure solver {status}. Aborting correction. !!!")
        raise RuntimeError(f"Step 3.2 Failure: PPE Solve did not converge at t={state.time}")

    # 3. CORRECT: Project V* onto a divergence-free space
    correct_velocity(state)

    # 4. HISTORY PERSISTENCE (SSoT Rule 4)
    # Removing getattr defaults. These properties MUST exist in state.health.
    state.history.times.append(state.time)
    
    # These assignments will raise an AttributeError if Step 3 didn't populate them.
    # This is preferred over silent 0.0 defaults (Zero-Debt).
    state.history.divergence_norms.append(state.health.divergence_norm)
    state.history.max_velocity_history.append(state.health.max_u)
    state.history.ppe_status_history.append(status)
    
    if DEBUG:
        print(f"DEBUG [Orchestrator]: Step 3 Complete. Final Div Norm: {state.health.divergence_norm:.2e}")

    # Flip the loop-readiness bit now that one full iteration is valid
    state.ready_for_time_loop = True 
    
    return state