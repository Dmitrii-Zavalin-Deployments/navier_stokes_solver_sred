# src/step1/orchestrate_step1.py

from __future__ import annotations

from typing import Any, Dict
import numpy as np
import json
import os
import jsonschema
from types import SimpleNamespace

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


# =====================================================================
# PURE DICT‑BASED ORCHESTRATOR (frozen Step‑1 contract)
# =====================================================================

def orchestrate_step1(
    json_input: Dict[str, Any],
    _unused_schema_argument: Dict[str, Any] = None,
    **_ignored_kwargs,
) -> Dict[str, Any]:
    """
    Step 1 — Minimal orchestrator aligned with the frozen Step‑1 schema.
    """

    # ---------------------------------------------------------
    # 0. Input schema validation (Fixes test_step1_input_schema_failure)
    # ---------------------------------------------------------
    # Enforces the immutable contract defined in solver_input_schema.json
    schema_path = os.path.join("schema", "solver_input_schema.json")
    try:
        with open(schema_path, "r") as f:
            input_schema = json.load(f)
        jsonschema.validate(instance=json_input, schema=input_schema)
    except (jsonschema.ValidationError, FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        # Re-raising as RuntimeError to satisfy the unit test expectation
        raise RuntimeError(f"Validation error: {exc}") from exc

    # ---------------------------------------------------------
    # 1. Parse config (dt + external_forces)
    # ---------------------------------------------------------
    config = parse_config(json_input)

    # ---------------------------------------------------------
    # 2. Validate physical constraints (Fixes test_step1_physical_constraints_failure)
    # ---------------------------------------------------------
    # Semantic check for density > 0, viscosity > 0, etc.
    validate_physical_constraints(json_input)

    # ---------------------------------------------------------
    # 3. Grid (nx, ny, nz, dx, dy, dz)
    # ---------------------------------------------------------
    # Aligned to frozen Step 1 dummy (2x2x2)
    grid = initialize_grid(json_input["domain"])

    # ---------------------------------------------------------
    # 4. Allocate fields (P, U, V, W)
    # ---------------------------------------------------------
    fields = allocate_fields(grid)

    # ---------------------------------------------------------
    # 5. Geometry mask (Fixes test_step1_geometry_mask_mapping & happy_path)
    # ---------------------------------------------------------
    # Reshapes flat mask list to (nx, ny, nz) NumPy array
    mask = map_geometry_mask(json_input["mask"], json_input["domain"])
    
    is_fluid = (mask == 1)
    # Logic for boundary detection (usually mask values <= 0 or specific flags)
    is_boundary_cell = (mask == -1)

    # ---------------------------------------------------------
    # 6. Derived constants (rho, mu, dt, dx, etc.)
    # ---------------------------------------------------------
    constants = compute_derived_constants(json_input)

    # ---------------------------------------------------------
    # 7. Boundary conditions (Fixes test_step1_boundary_conditions)
    # ---------------------------------------------------------
    # Ensures a valid list is returned, satisfying len() checks in tests
    boundary_conditions = parse_boundary_conditions(json_input.get("boundary_conditions", []))

    # ---------------------------------------------------------
    # 8. Empty containers (matching frozen Step‑1 dummy)
    # ---------------------------------------------------------
    operators = {}
    ppe = {}
    health = {}

    # ---------------------------------------------------------
    # 9. Assemble final Step‑1 state dict
    # ---------------------------------------------------------
    state_dict = {
        "config": config,
        "grid": grid,
        "fields": fields,
        "mask": mask,
        "is_fluid": is_fluid,
        "is_boundary_cell": is_boundary_cell,
        "constants": constants,
        "boundary_conditions": boundary_conditions,
        "operators": operators,
        "ppe": ppe,
        "health": health,
    }

    if DEBUG_STEP1:
        debug_state_step1(state_dict)

    return state_dict


# =====================================================================
# STATE‑BASED ORCHESTRATOR (returns SolverState)
# =====================================================================

def orchestrate_step1_state(json_input: Dict[str, Any]) -> SolverState:
    """
    Modern Step‑1 orchestrator: returns a SolverState object.
    Provides dot-notation access required by tests (e.g., state.constants.rho)
    """
    state_dict = orchestrate_step1(json_input)
    # unpacks the dictionary into the SolverState constructor/assembler
    return assemble_simulation_state(**state_dict)