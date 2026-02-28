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
    print("\n" + "="*20 + " DEBUG: STEP-1 STATE SUMMARY " + "="*20)
    attrs = ["grid", "fields", "constants", "mask", "boundary_conditions"]
    for attr in attrs:
        value = getattr(state_obj, attr, None)
        print(f"\n• {attr}: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"    ndarray shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"    dict keys={list(value.keys())}")
    print("="*69 + "\n")

def orchestrate_step1(
    json_input: Dict[str, Any],
    **_ignored_kwargs,
) -> SolverState:
    """
    Main entry point for Step 1. Transforms raw JSON into a validated SolverState.
    
    Constitutional Role: The High Command.
    Compliance: Vertical Integrity & Zero-Debt Mandate.
    """
    # 0. Structural Validation (The Legal Contract)
    schema_path = os.path.join("schema", "solver_input_schema.json")
    try:
        with open(schema_path, "r") as f:
            input_schema = json.load(f)
        jsonschema.validate(instance=json_input, schema=input_schema)
    except (jsonschema.ValidationError, FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"Contract Violation: Input schema validation FAILED: {exc}") from exc

    # 1. Parsing & Grid Initialization (Spatial Governor)
    grid_params = json_input["grid"]
    grid = initialize_grid(grid_params)
    config = parse_config(json_input)

    # 2. Field Allocation (Memory Architect)
    fields = allocate_fields(grid)
    
    # 3. Apply Initial Conditions (Field Primer)
    apply_initial_conditions(fields, json_input["initial_conditions"])

    # 4. Mask & Boundary Processing (Topology Interpreter)
    # Refined to receive the derived logical masks directly
    mask, is_fluid, is_boundary_cell = map_geometry_mask(json_input["mask"], grid_params)
    bc_table = parse_boundary_conditions(json_input["boundary_conditions"], grid)

    # 5. Numerical Constants (Mathematical Translator)
    constants = compute_derived_constants(
        grid, 
        json_input["fluid_properties"], 
        json_input["simulation_parameters"]
    )

    # 6. Assemble the State Object (Synthesis Hub)
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

    # 7. Physical Validation (Final Logic Gate)
    validate_physical_constraints(state)

    if DEBUG_STEP1:
        debug_state_step1(state)

    return state

