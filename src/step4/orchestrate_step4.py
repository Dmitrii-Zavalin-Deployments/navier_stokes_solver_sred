# src/step4/orchestrate_step4.py

from src.solver_state import SolverState
from .ghost_manager import initialize_ghost_fields
from .boundary_filler import fill_ghost_boundaries
from .audit_diagnostics import run_preflight_audit

def orchestrate_step4(state: SolverState) -> SolverState:
    """
    Step 4 Orchestrator: Ghost Padding & Audit.
    Point 1: Accept Step 3 Output.
    Point 2: Allocate extended fields, fill ghosts, and run audit.
    Point 3: Finalize readiness and return.
    """
    # 1. Expand fields to include halos
    initialize_ghost_fields(state)
    
    # 2. Synchronize boundaries across ghosts
    fill_ghost_boundaries(state)
    
    # 3. Perform diagnostic audit
    run_preflight_audit(state)
    
    # 4. Final Lock: State is now ready for iterative cycling
    state.ready_for_time_loop = True
    
    return state