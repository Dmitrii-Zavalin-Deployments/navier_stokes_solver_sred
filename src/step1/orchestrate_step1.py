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

DEBUG_STEP1 = True

def debug_state_step1(state_obj: SolverState) -> None:
    """Prints a summary using the SolverState object attributes."""
    print("\n==================== DEBUG: STEP‑1 STATE SUMMARY ====================")
    # We iterate over the main attributes expected in SolverState
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
    Populates the SolverState progressively.
    """
    # 0. Structural Validation
    schema_path = os.path.join("schema", "solver_input_schema.json")
    try:
        with open(schema_path, "r") as f:
            input_schema = json.load(f)
        jsonschema.validate(instance=json_input, schema=input_schema)
    except (jsonschema.ValidationError, FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"Input schema validation FAILED: {exc}") from exc

    # 1. Grid & Config Parsing (Renamed from domain to grid)
    grid_params = json_input["grid"]
    grid = initialize_grid(grid_params)
    
    # Ensure all coordinate extents are present in the grid dict
    grid.update({
        "x_min": grid_params["x_min"],
        "x_max": grid_params["x_max"],
        "y_min": grid_params["y_min"],
        "y_max": grid_params["y_max"],
        "z_min": grid_params["z_min"],
        "z_max": grid_params["z_max"],
    })

    config = parse_config(json_input)
    config["geometry"] = json_input.get("geometry", {})
    config["initial_conditions"] = json_input["initial_conditions"]

    # 2. Field Allocation
    fields = allocate_fields(grid)
    
    # 3. Mask & Boundary Processing
    # Pass the grid sub-dict to the mask mapper
    mask = map_geometry_mask(json_input["mask"], grid_params)
    bc_table = parse_boundary_conditions(json_input["boundary_conditions"], grid)

    # 4. Numerical Constants
    constants = compute_derived_constants(
        grid, 
        json_input["fluid_properties"], 
        json_input["simulation_parameters"]
    )

    # 5. Pre-calculate Mask Semantics
    # Schema: 1=fluid, 0=solid, -1=boundary-fluid
    is_fluid = (mask == 1) | (mask == -1)
    is_boundary_cell = (mask == -1)

    # 6. Assemble the State Object
    state = assemble_simulation_state(
        config=config,
        grid=grid,
        fields=fields,
        mask=mask,
        constants=constants,
        boundary_conditions=bc_table if bc_table else {},
        is_fluid=is_fluid,
        is_boundary_cell=is_boundary_cell
    )

    # Add the solid flag for visualization/Step 5
    state.is_solid = (mask == 0)

    # 7. Physical Validation
    # FIXED: Pass the 'state' object itself, NOT state.__dict__
    validate_physical_constraints(state)

    if DEBUG_STEP1:
        debug_state_step1(state)

    return state

def orchestrate_step1_state(json_input: Dict[str, Any]) -> SolverState:
    """Helper to ensure we return the SolverState object."""
    return orchestrate_step1(json_input)