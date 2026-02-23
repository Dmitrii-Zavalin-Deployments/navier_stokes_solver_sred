# src/step1/assemble_simulation_state.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np
from src.solver_state import SolverState

def assemble_simulation_state(
    config: Dict[str, Any],
    grid: Dict[str, Any],
    fields: Dict[str, Any],
    mask: np.ndarray,
    constants: Dict[str, Any],
    boundary_conditions: Dict[str, Any],
    is_fluid: np.ndarray,
    is_boundary_cell: np.ndarray,
    **kwargs
) -> SolverState:
    """
    Assembles the SolverState and creates the 'Traceability Mappings' 
    required for the Phase E Data Audit.
    
    Compliance: Uses Dictionary Injection. Direct property assignment is 
    forbidden to prevent desynchronization (No-Setter Mandate).
    """

    # 1. Primary Object Initialization
    # Data is passed into the 'fields' dict, which properties will read.
    state = SolverState(
        config=config,
        grid=grid,
        fields=fields,
        mask=mask,
        constants=constants,
        boundary_conditions=boundary_conditions,
        is_fluid=is_fluid,
        is_boundary_cell=is_boundary_cell
    )

    # 2. Physics Mapping (Internal Shorthand -> Schema Names)
    # This solves the KeyError: 'density' in the Data Coverage Audit
    state.fluid_properties = {
        "density": constants.get("rho"),
        "viscosity": constants.get("mu")
    }

    # 3. Validation of Field Integrity
    # Instead of assigning to properties, we verify the keys exist in the dict
    required_fields = ["U", "V", "W", "P"]
    for f in required_fields:
        if f not in state.fields:
            raise KeyError(f"Genesis Error: Required field '{f}' missing from allocation.")

    # 4. Logical State
    state.ready_for_time_loop = kwargs.get("ready_for_time_loop", False)
    
    return state