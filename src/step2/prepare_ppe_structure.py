# src/step2/prepare_ppe_structure.py

from __future__ import annotations
from src.solver_state import SolverState

def prepare_ppe_structure(state: SolverState) -> None:
    """
    Finalizes the PPE dictionary by linking math operators, 
    physical constants, and user-defined solver settings.
    """
    # 1. Physics (from Input Schema)
    rho = state.constants['rho']
    dt = state.config['dt']
    
    # 2. Tuning (from config.json)
    settings = state.config.get("solver_settings", {})

    # 3. Assembly
    state.ppe = {
        # Math: The actual sparse matrix object
        "A": state.operators.get("laplacian"),
        
        # Physics: The scaling coefficient
        "rhs_coeff": rho / dt,
        
        # Tuning: User preferences with safe fallbacks
        "solver_type": settings.get("solver_type", "PCG"),
        "tolerance": settings.get("ppe_tolerance", 1e-6),
        "max_iterations": settings.get("ppe_max_iter", 1000),
        
        # Diagnostics: Start healthy, let the solver update this if needed
        "ppe_is_singular": False
    }