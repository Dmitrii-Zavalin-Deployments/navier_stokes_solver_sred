# src/step4/orchestrate_step4.py

from src.solver_state import SolverState
from src.step4.initialize_extended_fields import initialize_extended_fields
from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics

def orchestrate_step4(state: SolverState) -> SolverState:
    """
    Step‑4 orchestrator.
    Prepares the SolverState for repeated Step‑3 time stepping by:
      • allocating extended fields
      • applying boundary conditions to ghost layers
      • computing Step‑4 diagnostics
      • marking the state as ready for the time loop
    """

    # 1. Allocate extended fields
    initialize_extended_fields(state)

    # 2. Apply boundary conditions to extended fields
    apply_boundary_conditions(state)

    # 3. Compute Step‑4 diagnostics
    assemble_diagnostics(state)

    # 4. Mark state as ready for time loop
    state.ready_for_time_loop = True

    return state