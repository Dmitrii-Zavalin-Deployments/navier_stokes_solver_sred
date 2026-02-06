# src/step3/orchestrate_step3.py

import json
from jsonschema import validate, ValidationError

from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre
from src.step3.predict_velocity import predict_velocity
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.solve_pressure import solve_pressure
from src.step3.correct_velocity import correct_velocity
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.step3.update_health import update_health
from src.step3.log_step_diagnostics import log_step_diagnostics


# Load Step 3 output schema once at import time
with open("schemas/step3_output_schema.json", "r") as f:
    STEP3_SCHEMA = json.load(f)


def step3(state, current_time, step_index):
    """
    Full Step 3 projection time step.

    Steps:
      1. Pre-boundary conditions
      2. Predict velocity (U*, V*, W*)
      3. Build PPE RHS
      4. Solve pressure
      5. Correct velocity
      6. Post-boundary conditions
      7. Update health diagnostics
      8. Log diagnostics
      9. Validate final state against Step 3 schema
    """

    # 1
    apply_boundary_conditions_pre(state)

    # 2
    U_star, V_star, W_star = predict_velocity(state)

    # 3
    rhs = build_ppe_rhs(state, U_star, V_star, W_star)

    # 4
    P_new = solve_pressure(state, rhs)

    # 5
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_new)

    # 6
    apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new)

    # 7
    update_health(state)

    # 8
    log_step_diagnostics(state, current_time, step_index)

    # 9 — SCHEMA VALIDATION (audit-grade)
    try:
        validate(instance=state, schema=STEP3_SCHEMA)
    except ValidationError as e:
        raise RuntimeError(
            f"Step 3 output does NOT match step3_output_schema.json:\n{e.message}"
        )

    return state