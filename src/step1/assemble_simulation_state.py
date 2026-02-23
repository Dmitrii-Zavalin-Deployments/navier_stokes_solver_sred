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
    """

    # 1. Primary Object Initialization
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
    # This solves the KeyError: 'density'
    state.fluid_properties = {
        "density": constants.get("rho"),
        "viscosity": constants.get("mu")
    }

    # 3. Field Mapping (Dictionary -> Attributes)
    # This enables state.velocity_u[0,0,0] access in tests
    state.velocity_u = fields.get("U")
    state.velocity_v = fields.get("V")
    state.velocity_w = fields.get("W")
    state.pressure = fields.get("P")

    # 4. Logical State
    state.ready_for_time_loop = kwargs.get("ready_for_time_loop", False)
    
    return state