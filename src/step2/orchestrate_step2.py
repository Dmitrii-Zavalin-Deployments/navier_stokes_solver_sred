# src/step2/orchestrate_step2.py

from src.common.solver_state import SolverState
from src.step2.stencil_assembler import assemble_stencil_matrix, registry

# Rule 7: Granular Traceability
DEBUG = True

def orchestrate_step2(state: SolverState) -> SolverState:
    """
    Orchestrates the construction of the Stencil Matrix.
    """
    # Explicitly clear the registry at the start of orchestration 
    # to ensure identity integrity for this specific simulation run.
    registry.clear()
    
    if DEBUG:
        print(f"DEBUG [Step 2.0]: Orchestration Started")

    state.stencil_matrix = assemble_stencil_matrix(state)
    
    state.ready_for_time_loop = True
    
    if DEBUG:
        print(f"DEBUG [Step 2.0]: Orchestration Finalized.")
    
    return state