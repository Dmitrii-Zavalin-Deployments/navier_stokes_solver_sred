from .schema_validator import validate_input_schema
from .validate_physical_constraints import validate_physical_constraints
from .initialize_grid import initialize_grid
from .allocate_staggered_fields import allocate_staggered_fields
from .map_geometry_mask import map_geometry_mask
from .compute_derived_constants import compute_derived_constants
from .verify_staggered_shapes import verify_staggered_shapes
from .assemble_simulation_state import assemble_simulation_state


def construct_simulation_state(json_input: dict):
    # 1. Schema validation
    validate_input_schema(json_input)

    # 2. Physical constraints
    validate_physical_constraints(json_input)

    # 3. Grid
    grid = initialize_grid(json_input)

    # 4. Fields
    P, U, V, W = allocate_staggered_fields(json_input, grid)

    # 5. Geometry mask
    mask = map_geometry_mask(json_input, grid)

    # 6. Derived constants
    constants = compute_derived_constants(json_input, grid)

    # 7. Shape verification
    verify_staggered_shapes(P, U, V, W, mask, grid)

    # 8. Final assembly
    return assemble_simulation_state(
        json_input, grid, P, U, V, W, mask, constants
    )
