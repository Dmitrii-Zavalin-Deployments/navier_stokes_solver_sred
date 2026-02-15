# file: src/step3/orchestrate_step3.py

import numpy as np

from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre
from src.step3.predict_velocity import predict_velocity
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.solve_pressure import solve_pressure
from src.step3.correct_velocity import correct_velocity
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.step3.update_health import update_health
from src.step3.log_step_diagnostics import log_step_diagnostics

from src.solver_state import SolverState

DEBUG_STEP3 = True


def debug_state_step3(state):
    print("\n==================== DEBUG: STEP‑3 STATE SUMMARY ====================")
    for key, value in state.items():
        print(f"\n• {key}: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"    ndarray shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"    dict keys={list(value.keys())}")
        elif hasattr(value, "__dict__"):
            print(f"    object attributes={list(vars(value).keys())}")
        else:
            print(f"    value={value}")
    print("====================================================================\n")


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

    # =====================================================================
    # DEPRECATED: per-step schema validation removed
    # =====================================================================

    base_state = dict(state)

    try:
        base_state = _ensure_is_solid(base_state)
    except Exception as exc:
        raise RuntimeError("[Step 3] Cannot infer is_solid") from exc

    try:
        fields0 = base_state["fields"]
    except KeyError:
        raise RuntimeError(
            "[Step 3] Input violates Step‑2 schema: missing required key 'fields'"
        )

    fields_pre = apply_boundary_conditions_pre(base_state, fields0)

    U_star, V_star, W_star = predict_velocity(base_state, fields_pre)

    rhs = build_ppe_rhs(base_state, U_star, V_star, W_star)

    P_arr, _ppe_meta = solve_pressure(base_state, rhs)

    U_new, V_new, W_new = correct_velocity(
        base_state, U_star, V_star, W_star, P_arr
    )

    fields_post = apply_boundary_conditions_post(
        base_state, U_new, V_new, W_new, P_arr
    )

    fields_out = dict(fields_post)
    fields_out["P"] = P_arr

    health = update_health(base_state, fields_out, P_arr)

    new_state = dict(base_state)
    new_state["fields"] = fields_out
    new_state["health"] = health

    diag_record = log_step_diagnostics(
        new_state, new_state["fields"], current_time, step_index
    )

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

    if DEBUG_STEP3:
        debug_state_step3(new_state)

    return new_state


def orchestrate_step3_state(
    state: SolverState,
    current_time: float,
    step_index: int,
) -> SolverState:

    state_dict = {
        "config": state.config,
        "grid": state.grid,
        "fields": state.fields,
        "mask": state.mask,
        "constants": state.constants,
        "boundary_conditions": state.boundary_conditions,
        "health": state.health,
        "ppe_structure": state.ppe,
        "operators": state.operators,
    }

    if getattr(state, "history", None):
        state_dict["history"] = state.history

    new_state_dict = orchestrate_step3(
        state_dict,
        current_time=current_time,
        step_index=step_index,
    )

    state.fields = new_state_dict["fields"]
    state.health = new_state_dict["health"]
    state.history = new_state_dict.get("history", {})

    return state
