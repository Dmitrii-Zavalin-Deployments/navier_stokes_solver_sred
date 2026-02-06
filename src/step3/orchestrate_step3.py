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
# Helper: Convert NumPy arrays → Python lists for JSON Schema validation
# ---------------------------------------------------------------------------

def _to_json_safe(obj):
    """
    Recursively convert numpy arrays to Python lists so JSON Schema can validate them.
    Functions and other non-JSON types are converted to simple placeholders.
    """
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]

    if callable(obj):
        return {}

    return obj


# ---------------------------------------------------------------------------
# Load schemas
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

STEP2_SCHEMA_PATH = os.path.join(ROOT, "schema", "step2_output_schema.json")
STEP3_SCHEMA_PATH = os.path.join(ROOT, "schema", "step3_output_schema.json")

with open(STEP2_SCHEMA_PATH, "r") as f:
    STEP2_SCHEMA = json.load(f)

with open(STEP3_SCHEMA_PATH, "r") as f:
    STEP3_SCHEMA = json.load(f)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def step3(state, current_time, step_index):
    """
    Full Step 3 projection time step.

    Steps:
      0. Validate input schema (must match Step 2 output)
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

    # ----------------------------------------------------------------------
    # 0 — INPUT SCHEMA VALIDATION (hard failure)
    # ----------------------------------------------------------------------
    try:
        json_safe_input = _to_json_safe(state)
        validate(instance=json_safe_input, schema=STEP2_SCHEMA)
    except ValidationError as exc:
        raise RuntimeError(
            f"\n[Step 3] Input schema validation FAILED.\n"
            f"Expected schema: {STEP2_SCHEMA_PATH}\n"
            f"Validation error: {exc.message}\n"
            f"Aborting Step 3 — upstream Step 2 output is malformed.\n"
        ) from exc

    # ----------------------------------------------------------------------
    # 1 — Pre-BCs
    # ----------------------------------------------------------------------
    apply_boundary_conditions_pre(state)

    # ----------------------------------------------------------------------
    # 2 — Predict velocity
    # ----------------------------------------------------------------------
    U_star, V_star, W_star = predict_velocity(state)

    # ----------------------------------------------------------------------
    # 3 — Build PPE RHS
    # ----------------------------------------------------------------------
    rhs = build_ppe_rhs(state, U_star, V_star, W_star)

    # ----------------------------------------------------------------------
    # 4 — Solve pressure
    # ----------------------------------------------------------------------
    P_new = solve_pressure(state, rhs)

    # ----------------------------------------------------------------------
    # 5 — Correct velocity
    # ----------------------------------------------------------------------
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_new)

    # ----------------------------------------------------------------------
    # 6 — Post-BCs
    # ----------------------------------------------------------------------
    apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new)

    # ----------------------------------------------------------------------
    # 7 — Update health
    # ----------------------------------------------------------------------
    update_health(state)

    # ----------------------------------------------------------------------
    # 8 — Log diagnostics
    # ----------------------------------------------------------------------
    log_step_diagnostics(state, current_time, step_index)

    # ----------------------------------------------------------------------
    # 9 — OUTPUT SCHEMA VALIDATION (hard failure)
    # ----------------------------------------------------------------------
    try:
        json_safe_state = _to_json_safe(state)
        validate(instance=json_safe_state, schema=STEP3_SCHEMA)
    except ValidationError as exc:
        raise RuntimeError(
            f"\n[Step 3] Output schema validation FAILED.\n"
            f"Expected schema: {STEP3_SCHEMA_PATH}\n"
            f"Validation error: {exc.message}\n"
            f"Aborting — Step 3 produced an invalid SimulationState.\n"
        ) from exc

    return state