# file: src/step1/orchestrate_step1.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np
from pathlib import Path

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


def _to_json_safe(obj):
    """Recursively convert NumPy arrays to Python lists for JSON/schema compatibility."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    return obj


def orchestrate_step1(
    json_input: Dict[str, Any],
    *,
    validate_json_schema=None,
    load_schema=None,
) -> Dict[str, Any]:
    """
    Orchestrate Step 1 of the solver pipeline.

    Responsibilities:
    - Validate input schema
    - Validate physical constraints
    - Parse config
    - Initialize grid
    - Allocate staggered fields
    - Map geometry mask
    - Apply initial conditions
    - Parse boundary conditions
    - Compute derived constants
    - Assemble final Step 1 state
    - Verify shapes
    - Validate output schema
    """

    # ---------------------------------------------------------
    # 1. Validate input JSON (original Step 1 logic)
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
    # 13. Create JSON‑safe mirror
    # ---------------------------------------------------------
    json_safe_state = {
        **state_dict,
        "fields": _to_json_safe(state_dict["fields"]),
        "mask_3d": _to_json_safe(state_dict["mask_3d"]),
    }

    # ---------------------------------------------------------
    # 14. Validate output schema (original Step 1 logic)
    # ---------------------------------------------------------
    output_schema = load_schema("schema/step1_output_schema.json")
    try:
        validate_with_schema(json_safe_state, output_schema)
    except Exception as exc:
        raise RuntimeError(
            "\n[Step 1] Output schema validation FAILED.\n"
            f"Validation error: {exc}\n"
        ) from exc

    # ---------------------------------------------------------
    # 15. Attach JSON‑safe mirror for serialization tests
    # ---------------------------------------------------------
    state_dict["state_as_dict"] = json_safe_state

    # ---------------------------------------------------------
    # 16. Return REAL state (NumPy arrays)
    # ---------------------------------------------------------
    return state_dict
