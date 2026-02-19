# src/step2/build_advection_structure.py

from __future__ import annotations
from src.solver_state import SolverState

def build_advection_structure(state: SolverState) -> None:
    """
    Prepares the advection data structures.
    
    Instead of calculating advection here, we build the metadata 
    and constants required for the non-linear term (u · ∇)u 
    to be computed efficiently in Step 3.
    """
    grid = state.grid
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    
    # Scale Guard: Accessing via dict keys
    dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
    
    # Retrieve scheme from config dict (defaulting to central if not set)
    # Using .get() because state.config is a dictionary
    scheme = state.config.get("advection_scheme", "central_difference_2nd_order")
    
    # This dictionary will be used by the Predictor in Step 3
    state.operators["advection"] = {
        "scheme": scheme,
        "grid_spacing": {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "inv_2dx": 1.0 / (2.0 * dx),
            "inv_2dy": 1.0 / (2.0 * dy),
            "inv_2dz": 1.0 / (2.0 * dz)
        },
        "is_fluid": state.is_fluid
    }