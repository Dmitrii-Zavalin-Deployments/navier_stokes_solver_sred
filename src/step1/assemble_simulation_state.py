# src/step1/assemble_simulation_state.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np

# REMOVED: from .types import (...)  <-- Fixing the ModuleNotFoundError

def assemble_simulation_state(
    config: Dict[str, Any],
    grid: Dict[str, Any],
    fields: Dict[str, Any],
    mask: np.ndarray,
    is_fluid: np.ndarray,
    is_boundary_cell: np.ndarray,
    constants: Dict[str, Any],
    boundary_conditions: Any = None,
    operators: Dict[str, Any] = None,
    ppe: Dict[str, Any] = None,
    health: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Assemble a solver-ready Step 1 state dictionary.

    This version removes dependencies on the missing '.types' module and 
    strictly follows the frozen 'solver_step1_output_dummy' structure.
    """

    # 1. Initialize empty containers for keys Step 1 doesn't fill yet
    # but are required by the output schema.
    if operators is None: operators = {}
    if ppe is None: ppe = {}
    if health is None: health = {}

    # 2. Assemble final Step 1 state dictionary
    # Keys match EXACTLY with EXPECTED_STEP1_SCHEMA and solver_output_schema.json
    state = {
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
        "ready_for_time_loop": False,
        "P_ext": None,
        "U_ext": None,
        "V_ext": None,
        "W_ext": None,
    }

    return state