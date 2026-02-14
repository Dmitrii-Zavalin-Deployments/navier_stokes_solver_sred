# src/step4/orchestrate_step4.py

from src.step4.initialize_extended_fields import initialize_extended_fields
from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics


def orchestrate_step4(
    state,
    validate_json_schema=None,
    load_schema=None,
):
    """
    Step 4 orchestrator.

    Responsibilities (simplified, production-oriented):
    - Initialize extended staggered fields (including mask semantics).
    - Enforce boundary conditions on pressure and velocity.
    - Assemble diagnostics (total_fluid_cells, max velocity, divergence, BC violations).
    - Mark the state as ready for the time loop.

    Schema validation hooks (validate_json_schema, load_schema) are accepted
    for compatibility but are not enforced here; they can be wired in at a
    higher level if desired.
    """

    # 1) Initialize extended fields (allocate + fill + mask semantics)
    state = initialize_extended_fields(state)

    # 2) Apply all boundary conditions (velocity + pressure)
    state = apply_boundary_conditions(state)

    # 3) Assemble diagnostics
    state = assemble_diagnostics(state)

    # 4) Mark ready for time loop
    state["ready_for_time_loop"] = True

    return state
