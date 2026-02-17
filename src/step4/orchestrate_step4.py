# src/step4/orchestrate_step4.py

from src.solver_state import SolverState
from src.step4.initialize_extended_fields import initialize_extended_fields
from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics


def orchestrate_step4_state(state: SolverState) -> SolverState:
    """
    Step‑4 orchestrator.
    Prepares the SolverState for repeated Step‑3 time stepping by:
      • allocating extended fields
      • applying boundary conditions to ghost layers
      • computing Step‑4 diagnostics
      • marking the state as ready for the time loop
    """

    # ---------------------------------------------------------
    # 1. Allocate extended fields
    # ---------------------------------------------------------
    initialize_extended_fields(state)

    # ---------------------------------------------------------
    # 2. Apply boundary conditions to extended fields
    # ---------------------------------------------------------
    apply_boundary_conditions(state)

    # ---------------------------------------------------------
    # 3. Compute Step‑4 diagnostics
    # ---------------------------------------------------------
    assemble_diagnostics(state)

    # ---------------------------------------------------------
    # 4. Mark state as ready for time loop
    # ---------------------------------------------------------
    state.ready_for_time_loop = True

    return state


# ---------------------------------------------------------------------------
# Legacy adapter (kept only for backward compatibility)
# ---------------------------------------------------------------------------

def orchestrate_step4(state_dict):
    """
    Legacy adapter for dict‑based pipelines.
    Converts a dict into a SolverState, runs Step‑4, and writes results back.
    """

    required = ["config", "fields", "is_fluid", "health"]
    for key in required:
        if key not in state_dict:
            raise ValueError(f"Missing required key '{key}' for Step‑4 adapter")

    # Build SolverState
    state = SolverState()
    state.config = state_dict["config"]
    state.fields = state_dict["fields"]
    state.is_fluid = state_dict["is_fluid"]
    state.health = state_dict["health"]

    # Run Step‑4
    orchestrate_step4_state(state)

    # Write back results
    state_dict["P_ext"] = state.P_ext
    state_dict["U_ext"] = state.U_ext
    state_dict["V_ext"] = state.V_ext
    state_dict["W_ext"] = state.W_ext
    state_dict["step4_diagnostics"] = state.step4_diagnostics
    state_dict["ready_for_time_loop"] = state.ready_for_time_loop

    return state_dict
