# src/step1/assemble_simulation_state.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np
from src.solver_state import SolverState

def assemble_simulation_state(
    config: Dict[str, Any],
    grid: Dict[str, Any],
    fields: Dict[str, np.ndarray],
    mask: list,
    constants: Dict[str, Any],
    boundary_conditions: Dict[str, Any],
    is_fluid: np.ndarray,
    is_boundary_cell: np.ndarray,
    **kwargs
) -> SolverState:
    """
    Unifies all components into the SolverState using manual hydration 
    to respect the dataclass constructor limitations.
    """

    # 1. Primary Object Initialization 
    # Only pass what the dataclass explicitly expects in its header
    state = SolverState(
        config=config,
        grid=grid,
        fields=fields
    )

    # 2. Manual Hydration (Direct Attribute Assignment)
    # This bypasses the __init__ and puts data directly into the 'Departments'
    state.constants = constants
    state.boundary_conditions = boundary_conditions
    
    # Populate the masks department (ensuring NumPy types)
    state.masks.mask = np.array(mask)
    state.masks.is_fluid = is_fluid
    state.masks.is_boundary_cell = is_boundary_cell

    # 3. Physics Mapping (Internal Shorthand)
    # Mapping rho/mu to density/viscosity for the health checks
    state.fluid_properties = {
        "density": constants.get("rho") or constants.get("density"),
        "viscosity": constants.get("mu") or constants.get("viscosity")
    }

    # 4. Status Flag
    state.ready_for_time_loop = kwargs.get("ready_for_time_loop", False)
    
    return state