# src/step3/orchestrate_step3.py

from src.solver_state import SolverState
from .predictor import predict_velocity
from .solver import solve_pressure
from .corrector import correct_velocity

def orchestrate_step3(state: SolverState) -> SolverState:
    """
    Step 3 Orchestrator: Physical Integration.
    Alignment: Accepts solver_step2_output_dummy and matches solver_step3_output_dummy.
    """
    # 1. Execute Numerical Steps
    predict_velocity(state)
    status = solve_pressure(state)
    correct_velocity(state)

    # 2. Update Progression (Mandate 5: Explicit update)
    state.iteration += 1
    state.time += state.config.dt
    
    # 3. History Persistence (SSoT Rule 4)
    state.history.times.append(state.time)
    state.history.divergence_norms.append(state.health.divergence_norm)
    state.history.max_velocity_history.append(state.health.max_u)
    state.history.ppe_status_history.append(status)
    
    # Flip the loop-readiness bit now that one full iteration is valid
    state.ready_for_time_loop = True 
    
    return state