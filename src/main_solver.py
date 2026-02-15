# src/main_solver.py

from typing import Dict, Any
from src.solver_state import SolverState
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from src.step4.orchestrate_step4 import orchestrate_step4


def run_solver(config: Dict[str, Any]) -> SolverState:
    """
    Transitional high-level solver pipeline.
    Runs the existing dict-based step orchestrators and aggregates their
    outputs into a unified SolverState object.

    This preserves all existing behavior while introducing the new
    architecture safely and incrementally.
    """

    # ---------------------------------------------------------
    # Step 1 (dict-based)
    # ---------------------------------------------------------
    step1_out = orchestrate_step1(config)

    state = SolverState()
    state.config = config
    state.grid = step1_out["grid"]
    state.fields = step1_out["fields"]
    state.mask = step1_out["mask"]
    state.constants = step1_out["constants"]
    state.boundary_conditions = step1_out["boundary_conditions"]
    state.health = step1_out["health"]

    # ---------------------------------------------------------
    # Step 2 (dict-based)
    # ---------------------------------------------------------
    step2_out = orchestrate_step2(step1_out)
    state.operators = step2_out["operators"]
    state.ppe = step2_out["ppe_structure"]
    state.health = step2_out["health"]
    state.is_fluid = step2_out.get("is_fluid")
    state.is_boundary_cell = step2_out.get("is_boundary_cell")

    # ---------------------------------------------------------
    # Step 3 (dict-based)
    # ---------------------------------------------------------
    step3_out = orchestrate_step3(step2_out)
    state.fields = step3_out["fields"]
    state.health = step3_out["health"]
    state.step3_diagnostics = step3_out.get("diagnostics", {})

    # ---------------------------------------------------------
    # Step 4 (dict-based)
    # ---------------------------------------------------------
    step4_out = orchestrate_step4(step3_out)
    state.P_ext = step4_out["P_ext"]
    state.U_ext = step4_out["U_ext"]
    state.V_ext = step4_out["V_ext"]
    state.W_ext = step4_out["W_ext"]
    state.step4_diagnostics = step4_out.get("diagnostics", {})
    state.ready_for_time_loop = step4_out.get("ready_for_time_loop", False)

    # ---------------------------------------------------------
    # Step 5 (future)
    # ---------------------------------------------------------
    # state.step5_outputs = step5(state)

    # Optional debug validation (disabled during migration)
    # from src.common.final_schema_utils import validate_final_state
    # validate_final_state(state)

    return state
