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

validate_json_schema = None
load_schema = None


def _to_json_compatible(obj):
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


def _normalize_laplacians(state):
    """
    Ensure laplacian_u/v/w are callables.
    If they are dicts (dummy state), replace with zero-operators.
    """
    ops = dict(state.get("operators", {}))

    def wrap(op):
        if callable(op):
            return op
        if isinstance(op, dict) and callable(op.get("op")):
            return op["op"]
        def zero(arr):
            return np.zeros_like(arr)
        return zero

    for key in ("laplacian_u", "laplacian_v", "laplacian_w"):
        if key in ops:
            ops[key] = wrap(ops[key])

    new_state = dict(state)
    new_state["operators"] = ops
    return new_state


def orchestrate_step3(state, current_time, step_index):

    if validate_json_schema and load_schema:
        step2_schema = load_schema("step2_output_schema.json")
        validate_json_schema(
            instance=_to_json_compatible(state),
            schema=step2_schema,
            context_label="[Step 3] Input schema validation",
        )

    base_state = dict(state)

    # Ensure operators are callables
    base_state = _normalize_laplacians(base_state)

    fields0 = base_state["fields"]
    fields_pre = apply_boundary_conditions_pre(base_state, fields0)

    U_star, V_star, W_star = predict_velocity(base_state, fields_pre)

    rhs = build_ppe_rhs(base_state, U_star, V_star, W_star)

    P_new = solve_pressure(base_state, rhs)

    U_new, V_new, W_new = correct_velocity(base_state, U_star, V_star, W_star, P_new)

    fields_post = apply_boundary_conditions_post(
        base_state, U_new, V_new, W_new, P_new
    )

    fields_out = dict(fields_post)
    fields_out["P"] = P_new

    health = update_health(base_state, fields_out, P_new)

    diag_record = log_step_diagnostics(
        base_state, fields_out, current_time, step_index
    )

    new_state = dict(base_state)
    new_state["fields"] = fields_out
    new_state["health"] = health

    history = list(new_state.get("history", []))
    history.append(diag_record)
    new_state["history"] = history

    if validate_json_schema and load_schema:
        step3_schema = load_schema("step3_output_schema.json")
        validate_json_schema(
            instance=_to_json_compatible(new_state),
            schema=step3_schema,
            context_label="[Step 3] Output schema validation",
        )

    return new_state


def step3(state, current_time, step_index):
    return orchestrate_step3(state, current_time, step_index)
