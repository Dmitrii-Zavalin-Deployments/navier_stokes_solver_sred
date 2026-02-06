# src/step3/orchestrate_step3.py

import json
import os
from jsonschema import validate, ValidationError

from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre
from src.step3.predict_velocity import predict_velocity
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.solve_pressure import solve_pressure
from src.step3.correct_velocity import correct_velocity
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.step3.update_health import update_health
from src.step3.log_step_diagnostics import log_step_diagnostics


# ---------------------------------------------------------------------------
# Load Step 3 schema relative to this file's directory
# ---------------------------------------------------------------------------

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # go up from src/step3 to project root
    "schema",
    "step3_output_schema.json"
)

with open(SCHEMA_PATH, "r") as f:
    STEP3_SCHEMA = json.load(f)


def step3(state, current_time, step_index):
    """
    Full Step 3 projection time step.
    """

    apply_boundary_conditions_pre(state)
    U_star, V_star, W_star = predict_velocity(state)
    rhs = build_ppe_rhs(state, U_star, V_star, W_star)
    P_new = solve_pressure(state, rhs)
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_new)
    apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new)
    update_health(state)
    log_step_diagnostics(state, current_time, step_index)

    # Validate final state against schema
    try:
        validate(instance=state, schema=STEP3_SCHEMA)
    except ValidationError as e:
        raise RuntimeError(
            f"Step 3 output does NOT match step3_output_schema.json:\n{e.message}"
        )

    return state
