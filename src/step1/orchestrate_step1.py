# src/step1/orchestrate_step1.py

from __future__ import annotations
import os
import json
import jsonschema
from typing import Any, Dict
import numpy as np

from src.solver_state import SolverState
from .parse_config import parse_config
from .initialize_grid import initialize_grid
from .allocate_fields import allocate_fields
from .map_geometry_mask import map_geometry_mask
from .parse_boundary_conditions import parse_boundary_conditions
from .compute_derived_constants import compute_derived_constants
from .validate_physical_constraints import validate_physical_constraints
from .assemble_simulation_state import assemble_simulation_state
from .apply_initial_conditions import apply_initial_conditions

DEBUG_STEP1 = True

def debug_state_step1(state_obj: SolverState) -> None:
    """Prints a summary using the SolverState object attributes."""
    print("\n==================== DEBUG: STEP‑1 STATE SUMMARY ====================")
    attrs = ["grid", "fields", "constants", "mask", "boundary_conditions"]
    for attr in attrs:
        value = getattr(state_obj, attr, None)
        print(f"\n• {attr}: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"    ndarray shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"    dict keys={list(value.keys())}")
    print("====================================================================\n")

def orchestrate_step1(
    json_input: Dict[str, Any],
    **_ignored_kwargs,
) -> SolverState:
    """
    Step 1 — Orchestrator: Strictly aligned with the production schema.
    """
    # 0. Structural Validation
    schema_path = os.path.join("schema", "solver_input_schema.json")
    try:
        with open(schema_path, "r") as f:
            input_schema = json.load(f)
        jsonschema.validate(instance=json_input, schema=input_schema)
    except (jsonschema.ValidationError, FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"Input schema validation FAILED: {exc}") from exc

    # 1. Parsing & Grid Initialization
    grid_params = json_input["grid"]
    grid = initialize_grid(grid_params)
    
    # Sync grid extents
    grid.update({
        "x_min": grid_params["x_min"], "x_max": grid_params["x_max"],
        "y_min": grid_params["y_min"], "y_max": grid_params["y_max"],
        "z_min": grid_params["z_min"], "z_max": grid_params["z_max"],
    })

    config = parse_config(json_input)

    # 2. Field Allocation
    fields = allocate_fields(grid)
    
    # 3. Apply Initial Conditions (Crucial for Data Audit)
    # This "paints" the loud values onto the zeroed 'fields' arrays
    apply_initial_conditions(fields, json_input["initial_conditions"])

    # 4. Mask & Boundary Processing
    mask = map_geometry_mask(json_input["mask"], grid_params)
    bc_table = parse_boundary_conditions(json_input["boundary_conditions"], grid)

    # 5. Numerical Constants
    constants = compute_derived_constants(
        grid, 
        json_input["fluid_properties"], 
        json_input["simulation_parameters"]
    )

    # 6. Pre-calculate Mask Semantics
    is_fluid = (mask == 1) | (mask == -1)
    is_boundary_cell = (mask == -1)

    # 7. Assemble the State Object
    # assemble_simulation_state is responsible for mapping density/velocity_u
    state = assemble_simulation_state(
        state=SolverState(), # Initialize the blank container
        config=config,
        grid_data=grid,
        fields=fields,
        constants=constants,
        mask=mask,
        bcs=bc_table if bc_table else {}
    )

    # Add logical flags
    state.is_fluid = is_fluid
    state.is_boundary_cell = is_boundary_cell
    state.is_solid = (mask == 0)

    # 8. Physical Validation
    validate_physical_constraints(state)

    if DEBUG_STEP1:
        debug_state_step1(state)

    return state

def orchestrate_step1_state(json_input: Dict[str, Any]) -> SolverState:
    return orchestrate_step1(json_input)