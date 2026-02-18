# src/step1/orchestrate_step1.py

from __future__ import annotations

from typing import Any, Dict
import numpy as np

from src.solver_state import SolverState
from .parse_config import parse_config

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
    _unused_schema_argument: Dict[str, Any] = None,
    **_ignored_kwargs,
) -> Dict[str, Any]:
    """
    Step 1 — Minimal orchestrator aligned with the frozen Step 1 schema
    and the frozen Step 1 dummy.

    Produces:
      • config
      • grid
      • fields
      • mask
      • is_fluid
      • is_boundary_cell
      • constants
      • boundary_conditions
      • operators
      • ppe
      • health
    """

    # ---------------------------------------------------------
    # 1. Parse config (dt + external_forces)
    # ---------------------------------------------------------
    config = parse_config(json_input)

    # ---------------------------------------------------------
    # 2. Grid (simple uniform grid, matching Step 1 dummy)
    # ---------------------------------------------------------
    nx = json_input["domain"]["nx"]
    ny = json_input["domain"]["ny"]
    nz = json_input["domain"]["nz"]

    grid = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
    }

    # ---------------------------------------------------------
    # 3. Constants (matching Step 1 dummy)
    # ---------------------------------------------------------
    constants = {
        "rho": json_input["fluid_properties"]["density"],
        "mu": json_input["fluid_properties"]["viscosity"],
        "dt": config["dt"],
        "dx": grid["dx"],
        "dy": grid["dy"],
        "dz": grid["dz"],
    }

    # ---------------------------------------------------------
    # 4. Mask (3D array of ints)
    # ---------------------------------------------------------
    mask_list = json_input["mask"]
    mask = np.array(mask_list, dtype=int)

    is_fluid = mask == 1
    is_boundary_cell = np.zeros_like(mask, dtype=bool)

    # ---------------------------------------------------------
    # 5. Fields (matching Step 1 dummy)
    # ---------------------------------------------------------
    fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # ---------------------------------------------------------
    # 6. Boundary conditions (Step 1 dummy sets None)
    # ---------------------------------------------------------
    boundary_conditions = None

    # ---------------------------------------------------------
    # 7. Empty containers (matching Step 1 dummy)
    # ---------------------------------------------------------
    operators = {}
    ppe = {}
    health = {}

    # ---------------------------------------------------------
    # 8. Assemble final Step 1 state dict
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
# STATE‑BASED STEP 1 ORCHESTRATOR
# =====================================================================

def orchestrate_step1_state(json_input: Dict[str, Any]) -> SolverState:
    """
    Modern Step 1 orchestrator: returns a SolverState object.
    """

    state_dict = orchestrate_step1(json_input)

    state = SolverState(
        config=state_dict["config"],
        grid=state_dict["grid"],
        fields=state_dict["fields"],
        mask=state_dict["mask"],
        is_fluid=state_dict["is_fluid"],
        is_boundary_cell=state_dict["is_boundary_cell"],
        constants=state_dict["constants"],
        boundary_conditions=state_dict["boundary_conditions"],
        operators=state_dict["operators"],
        ppe=state_dict["ppe"],
        health=state_dict["health"],
    )

    return state
