# src/step4/audit_diagnostics.py

from src.solver_state import SolverState

def run_preflight_audit(state: SolverState) -> None:
    """
    Point 2: Calculate memory footprint and CFL safety.
    """
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    # Calculate approximate memory footprint (float64 = 8 bytes)
    # We account for the 4 main extended fields
    total_voxels = (nx+2)*(ny+2)*(nz+2) 
    state.diagnostics.memory_footprint_gb = (total_voxels * 8 * 4) / 1e9
    
    # CFL Limit based on current max velocity from Step 3 health
    max_u = state.health.max_u
    if max_u > 0:
        state.diagnostics.initial_cfl_dt = 0.5 * state.grid.dx / max_u
    else:
        # Rule 5: Explicit error/value. If no velocity, we use a safe small DT.
        state.diagnostics.initial_cfl_dt = state.config.ppe_tolerance