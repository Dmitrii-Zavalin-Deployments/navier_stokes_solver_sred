# src/step1/assemble_simulation_state.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np

# Import the actual class we want to create
from src.solver_state import SolverState

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
) -> SolverState:
    """
    Assemble the central SolverState object for the pipeline.
    
    This replaces the dictionary-only approach with a proper object
    that travels through Steps 1-5.
    """

    # 1. Initialize Step 1 specific state into the SolverState object
    state = SolverState(
        config=config,
        grid=grid,
        fields=fields,
        mask=mask,
        constants=constants,
        boundary_conditions=boundary_conditions if boundary_conditions is not None else {},
        is_fluid=is_fluid,
        is_boundary_cell=is_boundary_cell,
        operators=operators if operators is not None else {},
        ppe=ppe if ppe is not None else {},
        health=health if health is not None else {}
    )

    # 2. Set any additional flags or extended fields if provided via kwargs
    # This keeps the assembly flexible for future steps
    state.ready_for_time_loop = kwargs.get("ready_for_time_loop", False)
    
    return state