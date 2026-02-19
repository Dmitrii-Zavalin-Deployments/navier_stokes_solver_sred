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
) -> Dict[str, Any]:
    """
    Step 1 — Orchestrator strictly aligned with frozen dummy and test suite requirements.
    """
    # 0. Structural Validation against frozen external schema
    schema_path = os.path.join("schema", "solver_input_schema.json")
    try:
        with open(schema_path, "r") as f:
            input_schema = json.load(f)
        jsonschema.validate(instance=json_input, schema=input_schema)
    except (jsonschema.ValidationError, FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        # UPDATED: Added specific string to satisfy test_step1_input_schema_failure
        raise RuntimeError(f"Input schema validation FAILED. Validation error: {exc}") from exc

    # 1. Logic Components
    validate_physical_constraints(json_input)
    grid = initialize_grid(json_input["domain"])
    
    # 2. Config Assembly
    # parse_config handles dt and external_forces. 
    # We manually inject the rest to satisfy tests looking for 'geometry'.
    config = parse_config(json_input)
    config["geometry"] = json_input.get("geometry", {})
    config["initial_conditions"] = json_input.get("initial_conditions", {})
    config["boundary_conditions"] = json_input.get("boundary_conditions", [])
    
    # 3. Field Allocation (Staggered shapes)
    fields = allocate_fields(grid)
    
    # 4. Geometry (Using frozen map_geometry_mask.py)
    mask = map_geometry_mask(json_input["mask"], json_input["domain"])
    
    # 5. Derived Constants
    constants = compute_derived_constants(
        grid, 
        json_input["fluid_properties"], 
        json_input["simulation_parameters"]
    )

    # 6. Boundary Conditions (Validation only for Step 1)
    parse_boundary_conditions(json_input.get("boundary_conditions", []), grid)

    # 7. Assemble final Step 1 state
    state_dict = {
        "config": config,
        "grid": grid,
        "fields": fields,
        "mask": mask,
        "is_fluid": (mask == 1),
        "is_boundary_cell": np.zeros_like(mask, dtype=bool),
        "constants": constants,
        "boundary_conditions": None, 
        "operators": {},
        "ppe": {},
        "health": {},
    }

    if DEBUG_STEP1:
        debug_state_step1(state_dict)

    return state_dict

def orchestrate_step1_state(json_input: Dict[str, Any]) -> SolverState:
    """
    Returns a SolverState object as required by the state-based tests.
    """
    state_dict = orchestrate_step1(json_input)
    # Ensure your assemble_simulation_state accepts these keys
    return assemble_simulation_state(**state_dict)