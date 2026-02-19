# src/step2/build_advection_structure.py

from __future__ import annotations
import numpy as np
from src.solver_state import SolverState

def build_advection_structure(state: SolverState) -> None:
    """
    Prepares the advection data structures.
    
    Instead of calculating advection here, we build the metadata 
    and coordinate mapping required for the non-linear term (u · ∇)u 
    to be computed efficiently in Step 3.
    """
    # Fix AttributeError: Access grid and constants via dictionary keys
    grid = state.grid
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    dx, dy, dz = state.constants['dx'], state.constants['dy'], state.constants['dz']
    
    # We define the numerical scheme here (e.g., 2nd-order Central or Upwind)
    # This ensures Step 3 knows exactly which physics to apply.
    state.operators["advection"] = {
        "scheme": "central_difference_2nd_order",
        "scaling": {
            "inv_2dx": 1.0 / (2.0 * dx),
            "inv_2dy": 1.0 / (2.0 * dy),
            "inv_2dz": 1.0 / (2.0 * dz)
        },
        "is_fluid": state.is_fluid
    }
    
    # Note: We no longer store 'advection_u', 'advection_v', etc. 
    # to maintain a clean, unified operator schema.