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
from .types import SimulationState
from .validate_physical_constraints import validate_physical_constraints
from .verify_staggered_shapes import verify_staggered_shapes
from .schema_utils import load_schema, validate_with_schema


# -------------------------------------------------------------------------
# Convert SimulationState → JSON‑serializable dict for schema validation
# -------------------------------------------------------------------------
def _state_to_dict(state: SimulationState) -> dict:
    return {
        "config": {
            "domain": state.config.domain,
            "fluid": state.config.fluid,
            "simulation": state.config.simulation,
            "forces": state.config.forces,
            "boundary_conditions": state.config.boundary_conditions,
            "geometry_definition": state.config.geometry_definition,
        },
        "grid": {
            "nx": state.grid.nx,
            "ny": state.grid.ny,
            "nz": state.grid.nz,
            "dx": state.grid.dx,
            "dy": state.grid.dy,
            "dz": state.grid.dz,
            "x_min": state.grid.x_min,
            "y_min": state.grid.y_min,
            "z_min": state.grid.z_min,
            "x_max": state.grid.x_max,
            "y_max": state.grid.y_max,
            "z_max": state.grid.z_max,
        },
        "fields": {
            "P": state.fields.P.tolist(),
            "U": state.fields.U.tolist(),
            "V": state.fields.V.tolist(),
            "W": state.fields.W.tolist(),
            "Mask": state.fields.Mask.tolist(),
        },
        "mask_3d": state.mask_3d.tolist(),
        "boundary_table": state.boundary_table,
        "constants": {
            "rho": state.constants.rho,
            "mu": state.constants.mu,
            "dt": state.constants.dt,
            "dx": state.constants.dx,
            "dy": state.constants.dy,
            "dz": state.constants.dz,
            "inv_dx": state.constants.inv_dx,
            "inv_dy": state.constants.inv_dy,
            "inv_dz": state.constants.inv_dz,
            "inv_dx2": state.constants.inv_dx2,
            "inv_dy2": state.constants.inv_dy2,
            "inv_dz2": state.constants.inv_dz2,
        }
    }


# -------------------------------------------------------------------------
# Main Step 1 Orchestrator
# -------------------------------------------------------------------------
def construct_simulation_state(
    json_input: Dict[str, Any],
    _unused_schema_argument: Dict[str, Any] = None,
) -> SimulationState:
    """
    High‑level orchestrator for Step 1.
    Performs:
      • Input schema validation
      • Physical validation
      • Allocation and mapping
      • Boundary normalization
      • Derived constants
      • Output schema validation
    """

    # ---------------------------------------------------------
    # 1. Validate input JSON against schema/input_schema.json
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
    # 2. Physical constraints (fatal checks)
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
    # 6. Map geometry mask (structural only)
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
    # 8. Boundary conditions (structural normalization)
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
    # 10. Assemble final SimulationState
    # ---------------------------------------------------------
    state = assemble_simulation_state(
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
    verify_staggered_shapes(state)

    # ---------------------------------------------------------
    # 12. Validate output schema (schema/step1_output_schema.json)
    # ---------------------------------------------------------
    output_schema = load_schema("schema/step1_output_schema.json")
    state_dict = _state_to_dict(state)

    try:
        validate_with_schema(state_dict, output_schema)
    except Exception as exc:
        raise RuntimeError(
            "\n[Step 1] Output schema validation FAILED.\n"
            "Expected schema: schema/step1_output_schema.json\n"
            f"Validation error: {exc}\n"
            "Aborting — Step 1 produced an invalid SimulationState.\n"
        ) from exc

    return state