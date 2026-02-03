from .schema_validator import validate_input_schema
from .validate_physical_constraints import validate_physical_constraints
from .initialize_grid import initialize_grid
from .allocate_staggered_fields import allocate_staggered_fields
from .map_geometry_mask import map_geometry_mask
from .compute_derived_constants import compute_derived_constants
from .verify_staggered_shapes import verify_staggered_shapes
from .assemble_simulation_state import assemble_simulation_state
from .validate_output_schema import validate_output_schema


def construct_simulation_state(json_input: dict):
    """
    Step 1 orchestrator:
    - Validate raw JSON input against input_schema.json
    - Validate physical constraints
    - Initialize grid
    - Allocate staggered fields
    - Map geometry mask
    - Compute derived constants
    - Verify shapes
    - Assemble SimulationState
    - Validate output against step1_output_schema.json
    """

    # 1. Validate input JSON structure
    validate_input_schema(json_input)

    # 2. Validate physical constraints
    validate_physical_constraints(json_input)

    # 3. Initialize grid
    grid = initialize_grid(json_input)

    # 4. Allocate staggered fields
    P, U, V, W = allocate_staggered_fields(json_input, grid)

    # 5. Map geometry mask
    mask = map_geometry_mask(json_input, grid)

    # 6. Compute derived constants
    constants = compute_derived_constants(json_input, grid)

    # 7. Verify shapes of all fields and mask
    verify_staggered_shapes(P, U, V, W, mask, grid)

    # 8. Assemble SimulationState
    state = assemble_simulation_state(
        json_input, grid, P, U, V, W, mask, constants
    )

    # 9. Validate output against step1_output_schema.json
    validate_output_schema(state)

    return state
