# src/step2/prepare_ppe_structure.py

from __future__ import annotations
from src.solver_state import SolverState

def prepare_ppe_structure(state: SolverState) -> None:
    """
    Finalizes the PPE dictionary by linking math operators, 
    physical constants, and user-defined solver settings.
    
    This structure provides the Pressure Poisson Equation solver with everything
    needed to calculate the pressure field from the intermediate velocity divergence.
    """
    # 1. Physics (from Input Schema)
    rho = state.constants['rho']
    dt = state.config['dt']
    
    # Scale Guard: Explicitly reject non-physical time steps to prevent ZeroDivisionError
    if dt <= 0:
        raise ValueError(
            f"Invalid time step (dt={dt}). The Pressure Poisson Equation requires "
            "a positive, non-zero time step to scale the divergence term."
        )
    
    # 2. Tuning (from config.json)
    settings = state.config.get("solver_settings", {})

    # 3. Assembly
    state.ppe = {
        # Math: The actual sparse matrix object representing the Laplacian operator
        "A": state.operators.get("laplacian"),
        
        # Physics: The scaling coefficient (rho / dt) used in the RHS of the PPE
        "rhs_coeff": rho / dt,
        
        # Tuning: Numerical solver preferences with safe fallbacks
        "solver_type": settings.get("solver_type", "PCG"),
        "tolerance": settings.get("ppe_tolerance", 1e-6),
        "max_iterations": settings.get("ppe_max_iter", 1000),
        
        # Diagnostics: Status flags for numerical health
        "ppe_is_singular": False
    }