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

    # 1. Grid & Config Parsing
    grid = initialize_grid(json_input["domain"])
    
    # Ensure physical extents are preserved in the grid dict for validation
    grid.update({
        "x_min": json_input["domain"]["x_min"],
        "x_max": json_input["domain"]["x_max"],
        "y_min": json_input["domain"]["y_min"],
        "y_max": json_input["domain"]["y_max"],
        "z_min": json_input["domain"]["z_min"],
        "z_max": json_input["domain"]["z_max"],
    })

    config = parse_config(json_input)
    config["geometry"] = json_input.get("geometry", {})
    config["initial_conditions"] = json_input["initial_conditions"]

    # 2. Field Allocation
    fields = allocate_fields(grid)
    
    # 3. Mask & Boundary Processing
    mask = map_geometry_mask(json_input["mask"], json_input["domain"])
    bc_table = parse_boundary_conditions(json_input["boundary_conditions"], grid)

    # 4. Numerical Constants
    constants = compute_derived_constants(
        grid, 
        json_input["fluid_properties"], 
        json_input["simulation_parameters"]
    )

    # 5. Assemble the State Object
    # We pass the collected parts to assemble_simulation_state to get our SolverState
    state = assemble_simulation_state(
        config=config,
        grid=grid,
        fields=fields,
        mask=mask,
        constants=constants,
        boundary_conditions=bc_table if bc_table else {},
    )

    # 6. Mask Semantics (Schema: 1=fluid, 0=solid, -1=boundary-fluid)
    # logic: fluid and boundary cells are 'active' for physics
    state.is_fluid = (mask == 1) | (mask == -1)
    state.is_boundary_cell = (mask == -1)
    # Added explicit solid flag for completeness and visualization
    state.is_solid = (mask == 0)

    # 7. Physical Validation
    # We validate the underlying data dictionary
    validate_physical_constraints(state.__dict__)

    if DEBUG_STEP1:
        # We pass the __dict__ to the debugger to see all internal attributes
        debug_state_step1(state.__dict__)

    return state