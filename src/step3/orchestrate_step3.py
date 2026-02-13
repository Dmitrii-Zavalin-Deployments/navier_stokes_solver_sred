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

DEBUG_STEP3 = False


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


def _ensure_is_solid(state):
    if "is_solid" in state:
        return state

    if "mask" in state:
        mask_arr = np.asarray(state["mask"])
        is_solid = (mask_arr == 0)
    else:
        fields = state.get("fields", {})
        sample = fields.get("U")
        if sample is None:
            raise KeyError("Cannot infer is_solid: no mask or fields available")
        is_solid = np.zeros_like(sample, dtype=bool)

    new_state = dict(state)
    new_state["is_solid"] = is_solid
    return new_state


def orchestrate_step3(
    state,
    current_time,
    step_index,
    _unused_schema_argument=None,
    **_ignored_kwargs,
):
    """
    Step 3 — Pressure projection and velocity correction.
    """

    # ------------------------------------------------------------
    # 0 — Validate Step‑2 output
    # ------------------------------------------------------------
    if validate_json_schema and load_schema:
        try:
            step2_schema = load_schema("step2_output_schema.json")
            validate_json_schema(
                instance=_to_json_compatible(state),
                schema=step2_schema,
                context_label="[Step 3] Input schema validation",
            )
        except Exception as exc:
            raise RuntimeError(
                "\n[Step 3] Input schema validation FAILED.\n"
                f"Validation error: {exc}\n"
            ) from exc

    # Defensive shallow copy
    base_state = dict(state)

    # ------------------------------------------------------------
    # Ensure is_solid exists
    # ------------------------------------------------------------
    try:
        base_state = _ensure_is_solid(base_state)
    except Exception as exc:
        raise RuntimeError("[Step 3] Cannot infer is_solid") from exc

    # ------------------------------------------------------------
    # 1 — Pre‑BC
    # ------------------------------------------------------------
    try:
        fields0 = base_state["fields"]
    except KeyError:
        raise RuntimeError(
            "[Step 3] Input violates Step‑2 schema: missing required key 'fields'"
        )

    fields_pre = apply_boundary_conditions_pre(base_state, fields0)

    # ------------------------------------------------------------
    # 2 — Predict velocity
    # ------------------------------------------------------------
    U_star, V_star, W_star = predict_velocity(base_state, fields_pre)

    # ------------------------------------------------------------
    # 3 — PPE RHS
    # ------------------------------------------------------------
    rhs = build_ppe_rhs(base_state, U_star, V_star, W_star)

    # ------------------------------------------------------------
    # 4 — Solve pressure
    # ------------------------------------------------------------
    P_arr, _ppe_meta = solve_pressure(base_state, rhs)

    # ------------------------------------------------------------
    # 5 — Correct velocity
    # ------------------------------------------------------------
    U_new, V_new, W_new = correct_velocity(
        base_state, U_star, V_star, W_star, P_arr
    )

    # ------------------------------------------------------------
    # 6 — Post‑BC
    # ------------------------------------------------------------
    fields_post = apply_boundary_conditions_post(
        base_state, U_new, V_new, W_new, P_arr
    )

    fields_out = dict(fields_post)
    fields_out["P"] = P_arr

    # ------------------------------------------------------------
    # 7 — Health
    # ------------------------------------------------------------
    health = update_health(base_state, fields_out, P_arr)

    # ------------------------------------------------------------
    # 8 — Assemble Step‑3 state BEFORE diagnostics
    # ------------------------------------------------------------
    new_state = dict(base_state)
    new_state["fields"] = fields_out
    new_state["health"] = health

    # ------------------------------------------------------------
    # 9 — Diagnostics (must use new_state)
    # ------------------------------------------------------------
    diag_record = log_step_diagnostics(
        new_state, new_state["fields"], current_time, step_index
    )

    # ------------------------------------------------------------
    # 10 — History
    # ------------------------------------------------------------
    hist = dict(
        base_state.get(
            "history",
            {
                "times": [],
                "divergence_norms": [],
                "max_velocity_history": [],
                "ppe_iterations_history": [],
                "energy_history": [],
            },
        )
    )

    hist["times"].append(diag_record.get("time", current_time))
    hist["divergence_norms"].append(diag_record.get("divergence_norm", 0.0))
    hist["max_velocity_history"].append(diag_record.get("max_velocity", 0.0))
    hist["ppe_iterations_history"].append(diag_record.get("ppe_iterations", -1))
    hist["energy_history"].append(diag_record.get("energy", 0.0))

    new_state["history"] = hist

    # ------------------------------------------------------------
    # 11 — Output schema validation (tests only)
    # ------------------------------------------------------------
    if validate_json_schema and load_schema:
        try:
            step3_schema = load_schema("step3_output_schema.json")
            validate_json_schema(
                instance=_to_json_compatible(new_state),
                schema=step3_schema,
                context_label="[Step 3] Output schema validation",
            )
        except Exception as exc:
            raise RuntimeError(
                "\n[Step 3] Output schema validation FAILED.\n"
                f"Validation error: {exc}\n"
            ) from exc

    # ------------------------------------------------------------
    # 12 — Optional debug print
    # ------------------------------------------------------------
    if DEBUG_STEP3:
        print("\n[DEBUG] Step‑3 output keys:", list(new_state.keys()))

    return new_state


def step3(state, current_time, step_index):
    return orchestrate_step3(state, current_time, step_index)
