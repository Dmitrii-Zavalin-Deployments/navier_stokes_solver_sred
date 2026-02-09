# src/step3/orchestrate_step3.py

import numpy as np

from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre
from src.step3.predict_velocity import predict_velocity
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.solve_pressure import solve_pressure
from src.step3.correct_velocity import correct_velocity
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.step3.update_health import update_health
from src.step3.log_step_diagnostics import log_step_diagnostics


# Optional, injectable schema helpers (can be set to None in tests)
validate_json_schema = None
load_schema = None


def _to_json_compatible(obj):
    """
    Recursively convert numpy arrays to Python lists and functions/callables
    to simple string placeholders so JSON Schema can validate them.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {k: _to_json_compatible(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_json_compatible(x) for x in obj]

    if callable(obj):
        name = getattr(obj, "__name__", obj.__class__.__name__)
        return f"<function {name}>"

    return obj


def orchestrate_step3(state, current_time, step_index):
    """
    Full Step‑3 projection time step.

    Inputs:
        state        – Step‑2 output dict (schema‑validated upstream)
        current_time – float
        step_index   – int

    Returns:
        new_state – Step‑3 output dict (schema‑compliant)
    """

    # ------------------------------------------------------------------
    # 0 — INPUT SCHEMA VALIDATION (if helpers are available)
    # ------------------------------------------------------------------
    if validate_json_schema is not None and load_schema is not None:
        step2_schema = load_schema("step2_output_schema.json")
        json_safe_input = _to_json_compatible(state)
        validate_json_schema(
            instance=json_safe_input,
            schema=step2_schema,
            context_label="[Step 3] Input schema validation",
        )

    # Shallow copy base state so we never mutate the input
    base_state = dict(state)

    # ------------------------------------------------------------------
    # 1 — Pre‑boundary conditions (on fields)
    # ------------------------------------------------------------------
    fields0 = base_state["fields"]
    fields_pre = apply_boundary_conditions_pre(base_state, fields0)

    # ------------------------------------------------------------------
    # 2 — Predict velocity
    # ------------------------------------------------------------------
    U_star, V_star, W_star = predict_velocity(base_state, fields_pre)

    # ------------------------------------------------------------------
    # 3 — Build PPE RHS
    # ------------------------------------------------------------------
    rhs = build_ppe_rhs(base_state, U_star, V_star, W_star)

    # ------------------------------------------------------------------
    # 4 — Solve pressure
    # ------------------------------------------------------------------
    P_new = solve_pressure(base_state, rhs)

    # ------------------------------------------------------------------
    # 5 — Correct velocity
    # ------------------------------------------------------------------
    U_new, V_new, W_new = correct_velocity(
        base_state, U_star, V_star, W_star, P_new
    )

    # ------------------------------------------------------------------
    # 6 — Post‑boundary conditions
    # ------------------------------------------------------------------
    fields_post = apply_boundary_conditions_post(
        base_state,
        U_new,
        V_new,
        W_new,
        P_new,
    )

    # Ensure pressure is present in fields
    fields_out = dict(fields_post)
    fields_out["P"] = P_new

    # ------------------------------------------------------------------
    # 7 — Update health (pure)
    # ------------------------------------------------------------------
    health = update_health(base_state, fields_out, P_new)

    # ------------------------------------------------------------------
    # 8 — Log diagnostics (pure)
    # ------------------------------------------------------------------
    diag_record = log_step_diagnostics(
        base_state, fields_out, current_time, step_index
    )

    # ------------------------------------------------------------------
    # 9 — Assemble new Step‑3 state
    # ------------------------------------------------------------------
    new_state = dict(base_state)
    new_state["fields"] = fields_out
    new_state["health"] = health

    # History is an explicit list of diagnostic records
    history = list(new_state.get("history", []))
    history.append(diag_record)
    new_state["history"] = history

    # ------------------------------------------------------------------
    # 10 — OUTPUT SCHEMA VALIDATION (if helpers are available)
    # ------------------------------------------------------------------
    if validate_json_schema is not None and load_schema is not None:
        step3_schema = load_schema("step3_output_schema.json")
        json_safe_output = _to_json_compatible(new_state)
        validate_json_schema(
            instance=json_safe_output,
            schema=step3_schema,
            context_label="[Step 3] Output schema validation",
        )

    return new_state


# ----------------------------------------------------------------------
# Backwards‑compatible alias required by test suite
# ----------------------------------------------------------------------
def step3(state, current_time, step_index):
    """
    Alias for orchestrate_step3, matching the naming convention used
    in Step‑1 and Step‑2 tests and ensuring compatibility with
    test_step3_integration.py and contract tests.
    """
    return orchestrate_step3(state, current_time, step_index)
