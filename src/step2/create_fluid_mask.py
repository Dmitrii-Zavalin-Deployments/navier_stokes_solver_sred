# src/step2/create_fluid_mask.py

from __future__ import annotations
import numpy as np
from src.solver_state import SolverState

def create_fluid_mask(state: SolverState) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert integer mask into boolean masks for operator use.
    
    Since state.mask is an Optional[np.ndarray], we ensure it exists 
    before performing logical operations.
    """
    if state.mask is None:
        raise ValueError("Cannot create fluid mask: state.mask is None.")

    mask = np.array(state.mask)

    # Scale Guard: Reject non-integer masks to prevent numerical ambiguity
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Mask must be an integer array, but got {mask.dtype}")
    if not np.all(np.isin(mask, [-1, 0, 1])):
        raise ValueError("Mask values must be restricted to {-1, 0, 1}")

    # Logic: Fluid is anything that is NOT solid (values 1 and -1)
    state.is_fluid = (mask != 0)
    
    # Logic: Specifically identify the boundary-fluid interface (-1)
    state.is_boundary_cell = (mask == -1)
    
    # Logic: Identify solid cells (0)
    state.is_solid = (mask == 0)

    return state.is_fluid, state.is_boundary_cell