# src/step1/orchestrate_step1.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from src.common.json_safe import to_json_safe
from src.solver_state import SolverState

from .assemble_simulation_state import assemble_simulation_state
from .compute_derived_constants import compute_derived_constants
from .initialize_grid import initialize_grid
from .map_geometry_mask import map_geometry_mask
from .parse_boundary_conditions import parse_boundary_conditions
from .parse_config import parse_config
from .validate_physical_constraints import validate_physical_constraints
from .schema_utils import load_schema, validate_with_schema
from .allocate_fields import allocate_fields
from .verify_cell_centered_shapes import verify_cell_centered_shapes

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
    """
    Step 1 — Parse, validate, and initialize a solver-ready state (dict form).
    """

    # ---------------------------------------------------------
    # 1. Validate input JSON
    # ---------------------------------------------------------
    input_schema = load_schema("step1_input_schema.json")
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
    # 4. Grid (cell-centered)
    # ---------------------------------------------------------
    grid = initialize_grid(config.domain)

    # ---------------------------------------------------------
    # 5. Allocate cell-centered fields
    # ---------------------------------------------------------
    fields = allocate_fields(grid)

    # ---------------------------------------------------------
    # 6. Map geometry mask (structural only)
    # ---------------------------------------------------------
    geom = config.geometry
    mask_flat = geom["mask_flat"]
    shape = tuple(geom["mask_shape"])
    order_formula = geom["flattening_order"]

    mask = map_geometry_mask(mask_flat, shape, order_formula)
    fields.Mask[...] = mask

    # ---------------------------------------------------------
    # 7. Apply initial conditions
    # ---------------------------------------------------------
    from .apply_initial_conditions import apply_initial_conditions
    apply_initial_conditions(fields, json_input["initial_conditions"])

    # ---------------------------------------------------------
    # 8. Boundary conditions (normalized table)
    # ---------------------------------------------------------
    bc_table = parse_boundary_conditions(config.boundary_conditions, grid)

    # ---------------------------------------------------------
    # 9. Derived constants
    # ---------------------------------------------------------
    constants = compute_derived_constants(
        grid_config=grid,
        fluid_properties=config.fluid_properties,
        simulation_parameters=config.simulation_parameters,
    )

    # ---------------------------------------------------------
    # 10. Assemble final Step 1 state (dict)
    # ---------------------------------------------------------
    state_dict = assemble_simulation_state(
        config=config,
        grid=grid,
        fields=fields,
        mask=mask,
        bc_table=bc_table,
        constants=constants,
    )

    # ---------------------------------------------------------
    # 11. Shape & consistency verification
    # ---------------------------------------------------------
    verify_cell_centered_shapes(state_dict)

    # ---------------------------------------------------------
    # 12. JSON‑safe mirror for tests
    # ---------------------------------------------------------
    state_dict["state_as_dict"] = to_json_safe(state_dict)

    # ---------------------------------------------------------
    # 13. Optional debug print
    # ---------------------------------------------------------
    if DEBUG_STEP1:
        debug_state_step1(state_dict)

    return state_dict


# =====================================================================
# STATE‑BASED STEP 1 ORCHESTRATOR
# =====================================================================

def orchestrate_step1_state(json_input: Dict[str, Any]) -> SolverState:
    """
    Modern Step 1 orchestrator: returns a SolverState object.
    """

    state_dict = orchestrate_step1(json_input)

    required = ["config", "grid", "fields", "mask", "constants", "boundary_conditions"]
    for key in required:
        if key not in state_dict:
            raise ValueError(f"Missing required key '{key}' in Step 1 output")

    state = SolverState()
    state.config = state_dict["config"]
    state.grid = state_dict["grid"]
    state.fields = state_dict["fields"]
    state.mask = state_dict["mask"]
    state.constants = state_dict["constants"]
    state.boundary_conditions = state_dict["boundary_conditions"]
    state.health = {}

    return state
