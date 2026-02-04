# file: step1/construct_simulation_state.py
from __future__ import annotations

from typing import Any, Dict


from .assemble_simulation_state import assemble_simulation_state
from .allocate_staggered_fields import allocate_staggered_fields
from .compute_derived_constants import compute_derived_constants
from .initialize_grid import initialize_grid
from .map_geometry_mask import map_geometry_mask
from .parse_boundary_conditions import parse_boundary_conditions
from .parse_config import parse_config
from .types import SimulationState
from .validate_json_schema import validate_json_schema
from .validate_physical_constraints import validate_physical_constraints
from .verify_staggered_shapes import verify_staggered_shapes


def construct_simulation_state(
    json_input: Dict[str, Any],
    schema: Dict[str, Any],
) -> SimulationState:
    """
    High-level orchestrator for Step 1.
    Structural + basic physical validation, allocation, and assembly.
    """

    # 1. Schema validation (structural)
    validate_json_schema(json_input, schema)

    # 2. Physical constraints (fatal checks)
    validate_physical_constraints(json_input)

    # 3. Parse config
    config = parse_config(json_input)

    # 4. Grid
    grid = initialize_grid(config.domain)

    # 5. Allocate fields
    fields = allocate_staggered_fields(grid)

    # 6. Map geometry mask (structural only, opaque integers)
    geom = config.geometry_definition
    mask_flat = geom["geometry_mask_flat"]
    shape = tuple(geom["geometry_mask_shape"])
    order_formula = geom["flattening_order"]
    mask_3d = map_geometry_mask(mask_flat, shape, order_formula)
    fields.Mask[...] = mask_3d

    # 7. Apply initial conditions
    apply_init = json_input["initial_conditions"]
    from .apply_initial_conditions import apply_initial_conditions
    apply_initial_conditions(fields, apply_init)

    # 8. Boundary conditions (structural normalization)
    bc_table = parse_boundary_conditions(config.boundary_conditions, grid)

    # 9. Derived constants
    constants = compute_derived_constants(
        grid_config=grid,
        fluid_properties=config.fluid,
        simulation_parameters=config.simulation,
    )

    # 10. Assemble state
    state = assemble_simulation_state(
        config=config,
        grid=grid,
        fields=fields,
        mask_3d=mask_3d,
        bc_table=bc_table,
        constants=constants,
    )

    # 11. Final shape verification
    verify_staggered_shapes(state)

    return state
