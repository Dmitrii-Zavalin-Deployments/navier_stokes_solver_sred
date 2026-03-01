# src/step5/orchestrate_step5_state.py

from src.solver_state import SolverState
from .archivist import record_snapshot
from .chronos_guard import synchronize_terminal_state

def orchestrate_step5(state: SolverState) -> SolverState:
    """
    Step 5 Orchestrator: Finalization & I/O.
    Point 1: Take Step 4 output.
    Point 2: Run Archivist (I/O) and Chronos Guard (Sync).
    Point 3: Return the "Gold Standard" state.
    """
    
    # 1. Trigger I/O (Snapshots and Manifest)
    # Only write if we hit the interval or the simulation is finished
    record_snapshot(state)
    
    # 2. Check for simulation completion
    synchronize_terminal_state(state)
    
    # 3. Final Checkpoint naming
    state.manifest.final_checkpoint = f"checkpoint_final_{state.iteration}.npy"
    
    return state