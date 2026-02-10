# src/step2/orchestrate_step2.py
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
import numpy as np

from .enforce_mask_semantics import enforce_mask_semantics
from .precompute_constants import precompute_constants
from .create_fluid_mask import create_fluid_mask
from .build_divergence_operator import build_divergence_operator
from .build_gradient_operators import build_gradient_operators
from .build_laplacian_operators import build_laplacian_operators
from .build_advection_structure import build_advection_structure
from .prepare_ppe_structure import prepare_ppe_structure
from .compute_initial_health import compute_initial_health

try:
    from ..step1.validate_json_schema import validate_json_schema
    from ..step1.schema_utils import load_schema
except Exception:
    validate_json_schema = None
    load_schema = None


# Only the fields that MUST exist for Step‑2 to run
REQUIRED_KEYS = ["grid", "fields", "mask_3d", "constants", "config"]


def orchestrate_step2(state: Dict[str, Any]) -> Dict[str, Any]:
    # Defensive copy
    state = deepcopy(state)

    # ---------------------------------------------------------
    # 0. Required‑key check (tests expect KeyError)
    # ---------------------------------------------------------
    for key in REQUIRED_KEYS:
        if key not in state:
            raise KeyError(f"Missing required Step‑1 field: '{key}'")

    # boundary_table is OPTIONAL — inject if missing
    if "boundary_table" not in state:
        state["boundary_table"] = {}

    # ---------------------------------------------------------
    # 1. Validate Step‑1 output (production safety)
    # ---------------------------------------------------------
    if validate_json_schema and load_schema:
        schema_path = (
            Path(__file__).resolve().parents[2] / "schema" / "step1_output_schema.json"
        )
        schema = load_schema(str(schema_path))
        try:
            validate_json_schema(state, schema)
        except Exception as exc:
            raise RuntimeError(
                "\n[Step 2] Input schema validation FAILED.\n"
                "The Step‑1 output does not match step1_output_schema.json.\n"
                f"Validation error: {exc}\n"
            ) from exc

    # ---------------------------------------------------------
    # 2. Precompute constants
    # ---------------------------------------------------------
    constants = precompute_constants(state)
    state["constants"] = constants

    # ---------------------------------------------------------
    # 3. Mask semantics
    # ---------------------------------------------------------
    mask_semantics = enforce_mask_semantics(state)

    # ---------------------------------------------------------
    # 4. Fluid mask
    # ---------------------------------------------------------
    is_fluid = create_fluid_mask(state)

    # ---------------------------------------------------------
    # 5. Compute is_solid
    # ---------------------------------------------------------
    mask_arr = np.asarray(state["mask_3d"])
    is_solid = (mask_arr == 0)

    # ---------------------------------------------------------
    # 6. Build operators
    # ---------------------------------------------------------
    _ = build_divergence_operator(state)
    _ = build_gradient_operators(state)
    _ = build_laplacian_operators(state)
    _ = build_advection_structure(state)

    # ---------------------------------------------------------
    # 7. PPE structure
    # ---------------------------------------------------------
    ppe = prepare_ppe_structure(state)

    # ---------------------------------------------------------
    # 8. Health diagnostics
    # ---------------------------------------------------------
    health = compute_initial_health(state)

    # ---------------------------------------------------------
    # 9. Assemble Step‑2 output
    # ---------------------------------------------------------
    output: Dict[str, Any] = {
        "grid": state["grid"],
        "fields": state["fields"],
        "config": state["config"],
        "constants": constants,
        "mask": state["mask_3d"],
        "is_fluid": is_fluid,
        "is_solid": is_solid.tolist(),
        "is_boundary_cell": mask_semantics["is_boundary_cell"],
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

    # ---------------------------------------------------------
    # 10. JSON‑safe PPE
    # ---------------------------------------------------------
    ppe_out = output.get("ppe", {})
    if "rhs_builder" in ppe_out and "rhs_builder_name" in ppe_out:
        ppe_out["rhs_builder"] = ppe_out["rhs_builder_name"]

    # ---------------------------------------------------------
    # 11. Validate Step‑2 output
    # ---------------------------------------------------------
    if validate_json_schema and load_schema:
        schema_path = (
            Path(__file__).resolve().parents[2] / "schema" / "step2_output_schema.json"
        )
        schema = load_schema(str(schema_path))
        try:
            validate_json_schema(output, schema)
        except Exception as exc:
            raise RuntimeError(
                "\n[Step 2] Output schema validation FAILED.\n"
                "The Step‑2 output does not match step2_output_schema.json.\n"
                f"Validation error: {exc}\n"
            ) from exc

    return output
