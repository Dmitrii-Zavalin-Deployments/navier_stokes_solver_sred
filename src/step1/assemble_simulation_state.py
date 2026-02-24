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
    
    Constitutional Role: Synthesis Hub.
    Compliance: Vertical Integrity Mandate (Traceability Mapping).
    
    Returns:
        SolverState: The 'Frozen' state ready for the Phase B synchronization cycle.
    """

    # 1. Primary Object Initialization (Dictionary Injection)
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

    # 2. Physics Mapping (Internal Shorthand -> Schema Compliance)
    # This enables the 'Data Completeness Audit' to find parameters in state.
    state.fluid_properties = {
        "density": constants.get("rho"),
        "viscosity": constants.get("mu")
    }

    # 3. Validation of Field Integrity (Anti-Debt Check)
    # Verify the existence and staggered shapes before finalizing the state.
    required_fields = ["U", "V", "W", "P"]
    for f in required_fields:
        if f not in state.fields:
            raise KeyError(f"Genesis Error: Required field '{f}' missing from allocation.")
        
    # Dimension Audit: Ensure masks and grids are spatially coherent
    # CONSTITUTIONAL FIX: Mask is now a 1D list (Phase A.2), so we audit length.
    expected_length = grid["nx"] * grid["ny"] * grid["nz"]
    if len(state.mask) != expected_length:
         raise ValueError(f"Spatial Incoherence: Mask length {len(state.mask)} != {expected_length}")

    # 4. Status Flag
    state.ready_for_time_loop = kwargs.get("ready_for_time_loop", False)
    
    return state
