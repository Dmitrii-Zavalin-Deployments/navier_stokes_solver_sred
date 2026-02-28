# src/step1/orchestrate_step1.py

from __future__ import annotations
import os
import json
import jsonschema
from typing import Any, Dict

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

# Constitutional Toggle: Set to False in production to avoid I/O overhead
DEBUG_STEP1 = True

def debug_state_step1(state_obj: SolverState) -> None:
    """
    Prints a summary using the SolverState object attributes.
    Ensures the SSoT containers are correctly populated.
    """
    print("\n" + "="*20 + " DEBUG: STEP-1 STATE SUMMARY " + "="*20)
    # We audit the key sub-containers as per SSoT Architecture
    containers = ["grid", "fields", "masks", "fluid", "config"]
    for attr in containers:
        value = getattr(state_obj, attr, None)
        print(f"\n• {attr.upper()}: {type(value)}")
        
        if attr == "fields" and value:
            print(f"    U shape: {value.U.shape}, V shape: {value.V.shape}, W shape: {value.W.shape}")
        elif attr == "grid" and value:
            print(f"    Topology: {value.nx}x{value.ny}x{value.nz}")
        elif attr == "fluid" and value:
            print(f"    Physics: rho={value.rho}, mu={value.mu}")
            
    print("\n" + "="*69 + "\n")

def orchestrate_step1(
    json_input: Dict[str, Any],
    **kwargs,
) -> SolverState:
    """
    Main entry point for Step 1. 
    Transforms raw JSON input into a validated, high-fidelity SolverState.
    
    Constitutional Role: The High Command.
    Mandate: Explicit Schema Validation before execution.
    """

    # 0. Contractual Gatekeeper (Structural Validation)
    # We check the input against the schema before any logic execution.
    schema_path = os.path.join("schema", "solver_input_schema.json")
    try:
        with open(schema_path, "r") as f:
            input_schema = json.load(f)
        jsonschema.validate(instance=json_input, schema=input_schema)
    except (jsonschema.ValidationError, FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        # Halt execution immediately if the input doesn't meet the legal requirement
        raise RuntimeError(f"Contract Violation: Input schema validation FAILED: {exc}") from exc

    # 1. Spatial Governor (Grid Context)
    grid_params = json_input["grid"]
    grid = initialize_grid(grid_params)
    
    # 2. Config Context (Solver Tuning)
    parse_config(json_input)

    # 3. Memory Architect (Staggered Field Allocation)
    fields = allocate_fields(grid)
    
    # 4. Field Primer (Initial Conditions)
    apply_initial_conditions(fields, json_input["initial_conditions"])

    # 5. Topology Interpreter (Masks & Boundaries)
    mask, is_fluid, is_boundary_cell = map_geometry_mask(json_input["mask"], grid_params)
    bc_table = parse_boundary_conditions(json_input["boundary_conditions"], grid)

    # 6. Mathematical Translator (Physical Constants)
    constants = compute_derived_constants(
        grid, 
        json_input["fluid_properties"], 
        json_input["simulation_parameters"]
    )

    # 7. Synthesis Hub (Assembly into SSoT Hierarchy)
    state = assemble_simulation_state(
        config_raw=json_input,
        grid_raw=grid,
        fields=fields,
        mask=mask,
        constants=constants,
        boundary_conditions=bc_table if bc_table else {},
        is_fluid=is_fluid,
        is_boundary_cell=is_boundary_cell,
        iteration=kwargs.get("iteration", 0),
        time=kwargs.get("time", 0.0),
        ready_for_time_loop=kwargs.get("ready_for_time_loop", False)
    )

    # 8. Logical Firewall (Final Physical Sanity Check)
    validate_physical_constraints(state)

    # 9. Debug Hook
    if DEBUG_STEP1:
        debug_state_step1(state)

    return state