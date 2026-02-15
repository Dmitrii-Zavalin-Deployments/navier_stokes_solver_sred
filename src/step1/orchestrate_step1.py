# file: src/step1/orchestrate_step1.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from src.common.json_safe import to_json_safe
from src.solver_state import SolverState  # NEW

from .assemble_simulation_state import assemble_simulation_state
from .allocate_staggered_fields import allocate_staggered_fields
from .compute_derived_constants import compute_derived_constants
from .initialize_grid import initialize_grid
from .map_geometry_mask import map_geometry_mask
from .parse_boundary_conditions import parse_boundary_conditions
from .parse_config import parse_config
from .validate_physical_constraints import validate_physical_constraints
from .verify_staggered_shapes import verify_staggered_shapes
from .schema_utils import load_schema, validate_with_schema


DEBUG_STEP1 = True


def debug_state_step1(state: Dict[str, Any]) -> None:
    print("\n==================== DEBUG: STEP‑1 STATE SUMMARY ====================")
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


def orchestrate_step1(
    json_input: Dict[str, Any],
    _unused_schema_argument: Dict[str, Any] = None,
    **_ignored_kwargs,
) -> Dict[str, Any]:

    # ---------------------------------------------------------
    # 1. Validate input JSON (KEEP)
    # ---------------------------------------------------------
    input_schema = load_schema("schema/input_schema.json")
    try:
        validate_with_schema(json_input, input_schema)
    except Exception as exc:
        raise RuntimeError(
            "\n[Step 1] Input schema validation FAILED.\n"
            f"Validation error: {exc}\n"
        ) from exc

    # ---------------------------------------------------------
    # 2. Physical constraints
    # ---------------------------------------------------------
    validate_physical_constraints(json_input)

    # ---------------------------------------------------------
    # 3. Parse config
    # ---------------------------------------------------------
    config = parse_config(json_input)

    # ---------------------------------------------------------
    # 4. Grid
    # ---------------------------------------------------------
    grid = initialize_grid(config.domain)

    # ---------------------------------------------------------
    # 5. Allocate staggered fields
    # ---------------------------------------------------------
    fields = allocate_staggered_fields(grid)

    # ---------------------------------------------------------
    # 6. Map geometry mask
    # ---------------------------------------------------------
    geom = config.geometry_definition
    mask_flat = geom["geometry_mask_flat"]
    shape = tuple(geom["geometry_mask_shape"])
    order_formula = geom["flattening_order"]

    mask_3d = map_geometry_mask(mask_flat, shape, order_formula)
    fields.Mask[...] = mask_3d

    # ---------------------------------------------------------
    # 7. Apply initial conditions
    # ---------------------------------------------------------
    from .apply_initial_conditions import apply_initial_conditions
    apply_initial_conditions(fields, json_input["initial_conditions"])

    # ---------------------------------------------------------
    # 8. Boundary conditions
    # ---------------------------------------------------------
    bc_table = parse_boundary_conditions(config.boundary_conditions, grid)

    # ---------------------------------------------------------
    # 9. Derived constants
    # ---------------------------------------------------------
    constants = compute_derived_constants(
        grid_config=grid,
        fluid_properties=config.fluid,
        simulation_parameters=config.simulation,
    )

    # ---------------------------------------------------------
    # 10. Assemble final Step 1 state
    # ---------------------------------------------------------
    state_dict = assemble_simulation_state(
        config=config,
        grid=grid,
        fields=fields,
        mask_3d=mask_3d,
        bc_table=bc_table,
        constants=constants,
    )

    # ---------------------------------------------------------
    # 11. Final shape verification
    # ---------------------------------------------------------
    verify_staggered_shapes(state_dict)

    # ---------------------------------------------------------
    # 12. Insert Mask into fields
    # ---------------------------------------------------------
    state_dict["fields"]["Mask"] = state_dict["mask_3d"]

    # ---------------------------------------------------------
    # 13. Create JSON‑safe mirror (KEEP — tests rely on this)
    # ---------------------------------------------------------
    json_safe_state = to_json_safe(state_dict)

    # =====================================================================
    # DEPRECATED: per-step output schema validation
    # Removed after full migration to SolverState + final_output_schema.json
    # =====================================================================
    # output_schema = load_schema("schema/step1_output_schema.json")
    # try:
    #     validate_with_schema(json_safe_state, output_schema)
    # except Exception as exc:
    #     raise RuntimeError(
    #         "\n[Step 1] Output schema validation FAILED.\n"
    #         f"Validation error: {exc}\n"
    #     ) from exc

    # ---------------------------------------------------------
    # 14. Attach JSON‑safe mirror for serialization tests
    # ---------------------------------------------------------
    state_dict["state_as_dict"] = json_safe_state

    # ---------------------------------------------------------
    # 15. Optional structured debug print
    # ---------------------------------------------------------
    if DEBUG_STEP1:
        debug_state_step1(state_dict)

    # ---------------------------------------------------------
    # 16. Return REAL state (NumPy arrays)
    # ---------------------------------------------------------
    return state_dict


# =====================================================================
# NEW: STATE‑BASED STEP 1 ORCHESTRATOR (incremental migration)
# =====================================================================

def orchestrate_step1_state(json_input: Dict[str, Any]) -> SolverState:
    """
    Modern Step 1 orchestrator: operates directly on SolverState.

    During migration, reuses the existing dict-based implementation
    by converting to/from dict internally.
    """

    state_dict = orchestrate_step1(json_input)

    required_keys = ["config", "grid", "fields", "mask_3d", "constants", "bc_table"]
    for key in required_keys:
        if key not in state_dict:
            raise ValueError(f"Missing required key '{key}' in Step 1 migration adapter")

    state = SolverState()
    state.config = state_dict["config"]
    state.grid = state_dict["grid"]
    state.fields = state_dict["fields"]
    state.mask = state_dict["mask_3d"]
    state.constants = state_dict["constants"]
    state.boundary_conditions = state_dict["bc_table"]
    state.health = {}  # Step 1 does not compute health

    return state
