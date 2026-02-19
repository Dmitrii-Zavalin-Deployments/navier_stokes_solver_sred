# src/step2/prepare_ppe_structure.py

from __future__ import annotations
import numpy as np
from src.solver_state import SolverState

def prepare_ppe_structure(state: SolverState) -> None:
    """
    Prepare PPE metadata and source term coefficients.
    
    This structure provides the necessary parameters for the Pressure 
    Poisson Equation (Lp = b), where the source term b is derived 
    from the velocity divergence.
    """

    # Fix AttributeError: Access constants and config via dictionary keys
    rho = state.constants['rho']
    dt = state.config['dt']
    
    # We store the physical scaling factor: (rho / dt)
    # This transforms the kinematic divergence into a dynamic pressure source.
    # We also include solver-specific parameters for the iterative PCG method.
    state.ppe = {
        "rhs_coeff": rho / dt,
        "solver_type": "PCG",
        "tolerance": 1e-6,
        "max_iterations": 1000,
        "ppe_is_singular": False,  # Updated by the solver if null-spaces are detected
    }