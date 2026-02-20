# src/main_solver.py

from typing import Dict, Any

from src.solver_state import SolverState
from src.step1.orchestrate_step1 import orchestrate_step1_state
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3_state
from src.step4.orchestrate_step4 import orchestrate_step4_state

from src.common.final_schema_utils import validate_final_state


def run_solver(config: Dict[str, Any]) -> SolverState:
    """
    High-level solver pipeline using the new state-based orchestrators.
    Produces a fully-populated SolverState and validates it against the
    final_output_schema.json.

    This is the unified entry point for the solver after migration.
    """

    # ---------------------------------------------------------
    # Step 1 — initialize SolverState
    # ---------------------------------------------------------
    state = orchestrate_step1_state(config)

    # ---------------------------------------------------------
    # Step 2 — numerical preprocessing
    # ---------------------------------------------------------
    state = orchestrate_step2(state)

    # ---------------------------------------------------------
    # Step 3 — pressure projection + velocity correction
    # ---------------------------------------------------------
    state = orchestrate_step3_state(
        state,
        current_time=0.0,   # placeholder until time loop is added
        step_index=0        # placeholder until time loop is added
    )

    # ---------------------------------------------------------
    # Step 4 — extended fields + diagnostics
    # ---------------------------------------------------------
    state = orchestrate_step4_state(state)

    # ---------------------------------------------------------
    # Final schema validation (Step 6)
    # ---------------------------------------------------------
    validate_final_state(state)

    return state
