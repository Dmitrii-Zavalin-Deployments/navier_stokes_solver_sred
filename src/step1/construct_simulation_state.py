# file: src/step1/construct_simulation_state.py
from __future__ import annotations

from typing import Any, Dict

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


def construct_simulation_state(
    json_input: Dict[str, Any],
    _unused_schema_argument: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Step 1 Orchestrator (schema‑aligned).

    Performs:
      • Input schema validation
      • Physical validation
      • Grid construction
      • Field allocation
      • Geometry mask mapping
      • Initial conditions
      • Boundary normalization
      • Derived constants
      • Output assembly
      • Output schema validation

    Returns:
        A dict matching the Step 1 Output Schema exactly.
    """

    # ---------------------------------------------------------
    # 1. Validate input JSON against Step 1 Input Schema
    # ---------------------------------------------------------
    input_schema = load_schema("schema/input_schema.json")
    try:
        validate_with_schema(json_input, input_schema)
    except Exception as exc:
        raise RuntimeError(
            "\n[Step 1] Input schema validation FAILED.\n"
            "Expected schema: schema/input_schema.json\n"
            f"Validation error: {exc}\n"
            "Aborting Step 1 — input JSON is malformed.\n"
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
    # 10. Assemble final Step 1 state (as a dict)
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
    # 12. Validate output schema
    # ---------------------------------------------------------
    output_schema = load_schema("schema/step1_output_schema.json")

    try:
        validate_with_schema(state_dict, output_schema)
    except Exception as exc:
        raise RuntimeError(
            "\n[Step 1] Output schema validation FAILED.\n"
            "Expected schema: schema/step1_output_schema.json\n"
            f"Validation error: {exc}\n"
            "Aborting — Step 1 produced an invalid SimulationState.\n"
        ) from exc

    return state_dict
