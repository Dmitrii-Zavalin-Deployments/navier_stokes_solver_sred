# file: src/step2/orchestrate_step2.py
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict
import numpy as np

from src.solver_state import SolverState  # NEW

from .enforce_mask_semantics import enforce_mask_semantics
from .precompute_constants import precompute_constants
from .create_fluid_mask import create_fluid_mask
from .build_divergence_operator import build_divergence_operator
from .build_gradient_operators import build_gradient_operators
from .build_laplacian_operators import build_laplacian_operators
from .build_advection_structure import build_advection_structure
from .prepare_ppe_structure import prepare_ppe_structure
from .compute_initial_health import compute_initial_health

DEBUG_STEP2 = True


def debug_state_step2(state: Dict[str, Any]) -> None:
    print("\n==================== DEBUG: STEP‑2 STATE SUMMARY ====================")
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


def _extract_gradients(gradients: Any) -> Any:
    if not isinstance(gradients, dict):
        return gradients
    if "pressure_gradients" not in gradients:
        return gradients

    pg = gradients["pressure_gradients"]
    if "x" in pg:
        return pg["x"], pg["y"], pg["z"]
    if "px" in pg:
        return pg["px"], pg["py"], pg["pz"]
    if "dpdx" in pg:
        return pg["dpdx"], pg["dpdy"], pg["dpdz"]
    if "gx" in pg:
        return pg["gx"], pg["gy"], pg["gz"]

    raise KeyError(
        "pressure_gradients must contain x/y/z, px/py/pz, dpdx/dpdy/dpdz, or gx/gy/gz"
    )


def orchestrate_step2(
    state: Dict[str, Any],
    _unused_schema_argument: Dict[str, Any] = None,
    **_ignored_kwargs,
) -> Dict[str, Any]:
    """
    Step 2 — Numerical preprocessing.
    """

    state = deepcopy(state)

    # =====================================================================
    # DEPRECATED: per-step schema validation removed
    # =====================================================================

    constants = precompute_constants(state)
    mask_semantics = enforce_mask_semantics(state)
    is_fluid, is_boundary_cell = create_fluid_mask(state)

    mask_arr = np.asarray(state["mask_3d"])
    is_solid = (mask_arr == 0)

    _ = build_divergence_operator(state)
    gradients = build_gradient_operators(state)
    _ = build_laplacian_operators(state)
    _ = build_advection_structure(state)

    _ = _extract_gradients(gradients)

    ppe = prepare_ppe_structure(state)

    health = compute_initial_health(
        {
            **state,
            "constants": constants,
        }
    )

    output: Dict[str, Any] = {
        "grid": state["grid"],
        "fields": state["fields"],
        "config": state["config"],
        "constants": constants,
        "mask": state["mask_3d"],
        "is_fluid": is_fluid.tolist(),
        "is_solid": is_solid.tolist(),
        "is_boundary_cell": is_boundary_cell.tolist(),
        "operators": {
            "divergence": "divergence",
            "gradient_p_x": "gradient_p_x",
            "gradient_p_y": "gradient_p_y",
            "gradient_p_z": "gradient_p_z",
            "laplacian_u": "laplacian_u",
            "laplacian_v": "laplacian_v",
            "laplacian_w": "laplacian_w",
            "advection_u": "advection_u",
            "advection_v": "advection_v",
            "advection_w": "advection_w",
        },
        "ppe": ppe,
        "ppe_structure": ppe,
        "health": health,
        "meta": {
            "step": 2,
            "description": "Step‑2 numerical preprocessing",
        },
    }

    ppe_out = output.get("ppe", {})
    if "rhs_builder" in ppe_out and "rhs_builder_name" in ppe_out:
        ppe_out["rhs_builder"] = ppe_out["rhs_builder_name"]

    if DEBUG_STEP2:
        debug_state_step2(output)

    return output


def orchestrate_step2_state(state: SolverState) -> SolverState:
    state_dict = {
        "config": state.config,
        "grid": state.grid,
        "fields": state.fields,
        "mask_3d": state.mask,
        "constants": state.constants,
        "boundary_conditions": state.boundary_conditions,
        "health": state.health,
    }

    new_state_dict = orchestrate_step2(state_dict)

    state.operators = new_state_dict["operators"]
    state.ppe = new_state_dict["ppe"]
    state.health = new_state_dict["health"]
    state.is_fluid = new_state_dict["is_fluid"]
    state.is_boundary_cell = new_state_dict["is_boundary_cell"]

    return state
