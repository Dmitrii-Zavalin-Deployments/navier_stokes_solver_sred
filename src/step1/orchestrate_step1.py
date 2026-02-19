# src/step1/orchestrate_step1.py

from __future__ import annotations

from typing import Any, Dict
import numpy as np
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
    # 0. Input schema validation (tests expect RuntimeError)
    # ---------------------------------------------------------
    required = [
        "domain",
        "fluid_properties",
        "initial_conditions",
        "boundary_conditions",
        "external_forces",
    ]
    for key in required:
        if key not in json_input:
            raise RuntimeError(f"Missing required key: {key}")

    # ---------------------------------------------------------
    # 1. Parse config (dt + external_forces)
    # ---------------------------------------------------------
    config = parse_config(json_input)

    # ---------------------------------------------------------
    # 2. Validate physical constraints (density, viscosity, etc.)
    # ---------------------------------------------------------
    validate_physical_constraints(json_input)

    # ---------------------------------------------------------
    # 3. Grid (nx, ny, nz, dx, dy, dz)
    # ---------------------------------------------------------
    grid = initialize_grid(json_input)

    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]

    # ---------------------------------------------------------
    # 4. Allocate fields (P, U, V, W)
    # ---------------------------------------------------------
    fields = allocate_fields(grid)

    # ---------------------------------------------------------
    # 5. Geometry mask (3‑D int array)
    # ---------------------------------------------------------
    mask = map_geometry_mask(json_input, grid)
    is_fluid = mask == 1
    is_boundary_cell = np.zeros_like(mask, dtype=bool)

    # ---------------------------------------------------------
    # 6. Derived constants (SimpleNamespace)
    # ---------------------------------------------------------
    constants = compute_derived_constants(json_input, grid, config)

    # ---------------------------------------------------------
    # 7. Boundary conditions (must be list, not None)
    # ---------------------------------------------------------
    boundary_conditions = parse_boundary_conditions(json_input)

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
    """

    state_dict = orchestrate_step1(json_input)
    return assemble_simulation_state(state_dict)
