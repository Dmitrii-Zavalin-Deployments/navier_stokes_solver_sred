# src/step4/orchestrate_step4.py

from typing import Dict, Any
from src.solver_state import SolverState
from src.step4.initialize_extended_fields import initialize_extended_fields
from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics


def orchestrate_step4_state(state: SolverState) -> SolverState:
    """
    Modern Step 4 orchestrator: operates directly on SolverState.
    """

    # =====================================================================
    # DEPRECATED: per-step schema validation
    # Step 4 historically had no schema validation blocks.
    # Documented here for consistency with Steps 1–3 after migration to
    # SolverState + final_output_schema.json.
    # =====================================================================

    extended = initialize_extended_fields(
        fields=state.fields,
        mask=state.mask,
        config=state.config
    )

    state.P_ext = extended["P_ext"]
    state.U_ext = extended["U_ext"]
    state.V_ext = extended["V_ext"]
    state.W_ext = extended["W_ext"]

    bc_result = apply_boundary_conditions(
        P_ext=state.P_ext,
        U_ext=state.U_ext,
        V_ext=state.V_ext,
        W_ext=state.W_ext,
        mask=state.mask,
        config=state.config,
        health=state.health,
    )

    state.P_ext = bc_result.get("P_ext", state.P_ext)
    state.U_ext = bc_result.get("U_ext", state.U_ext)
    state.V_ext = bc_result.get("V_ext", state.V_ext)
    state.W_ext = bc_result.get("W_ext", state.W_ext)
    state.health = bc_result.get("health", state.health)

    diagnostics = assemble_diagnostics(
        P_ext=state.P_ext,
        U_ext=state.U_ext,
        V_ext=state.V_ext,
        W_ext=state.W_ext,
        mask=state.mask,
        health=state.health,
        config=state.config,
    )
    state.step4_diagnostics = diagnostics

    state.ready_for_time_loop = True

    return state


def orchestrate_step4(state_dict: Dict[str, Any]) -> Dict[str, Any]:

    required_keys = ["config", "fields", "mask", "health"]
    for key in required_keys:
        if key not in state_dict:
            raise ValueError(f"Missing required key '{key}' for Step 4 adapter")

    state = SolverState()
    state.config = state_dict["config"]
    state.fields = state_dict["fields"]
    state.mask = state_dict["mask"]
    state.health = state_dict["health"]

    state = orchestrate_step4_state(state)

    state_dict["P_ext"] = state.P_ext
    state_dict["U_ext"] = state.U_ext
    state_dict["V_ext"] = state.V_ext
    state_dict["W_ext"] = state.W_ext
    state_dict["diagnostics"] = state.step4_diagnostics
    state_dict["ready_for_time_loop"] = state.ready_for_time_loop

    return state_dict
