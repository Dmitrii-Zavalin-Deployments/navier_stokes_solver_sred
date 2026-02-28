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
    Unifies all initialized components into the SolverState container.
    """

    # 1. Primary Object Initialization (Strict Signature Compliance)
    # Note: 'mask' is removed from the constructor to prevent TypeError
    state = SolverState(
        config=config,
        grid=grid,
        fields=fields,
        constants=constants,
        boundary_conditions=boundary_conditions,
        is_fluid=is_fluid,
        is_boundary_cell=is_boundary_cell
    )

    # 2. Attach Masking Data to the Correct Department
    # This prevents the 'mask' from floating as a top-level attribute.
    state.masks.mask = np.array(mask)

    # 3. Physics Mapping (Internal Shorthand -> Schema Compliance)
    state.fluid_properties = {
        "density": constants.get("rho"),
        "viscosity": constants.get("mu")
    }

    # 4. Dimension Audit
    expected_length = grid["nx"] * grid["ny"] * grid["nz"]
    if state.masks.mask.size != expected_length:
         raise ValueError(f"Spatial Incoherence: Mask size {state.masks.mask.size} != {expected_length}")

    # 5. Status Flag
    state.ready_for_time_loop = kwargs.get("ready_for_time_loop", False)
    
    return state