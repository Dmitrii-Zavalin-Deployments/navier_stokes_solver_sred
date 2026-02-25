# src/step2/prepare_ppe_structure.py

from __future__ import annotations
from src.solver_state import SolverState
import numpy as np

def prepare_ppe_structure(state: SolverState) -> None:
    """
    Finalizes the PPE dictionary by linking math operators, 
    physical constants, and user-defined solver settings.
    """
    # 1. Physics
    rho = state.constants['rho']
    dt = state.constants['dt']
    
    if dt <= 0:
        raise ValueError(f"Invalid time step (dt={dt}). Must be positive.")
    
    # 2. Singularity Check: 
    # The PPE is singular if there are no pressure (Dirichlet) outlets.
    # We check if "pressure_outlet" exists in the boundary_conditions dict.
    bc = state.boundary_conditions or {}
    has_outlet = any(b.get("type") == "pressure" for b in bc)
    is_singular = not has_outlet

    # 3. RHS Builder Function
    # The RHS of the PPE is: - (rho / dt) * divergence
    # We also mask it to ensure no pressure updates happen inside solid cells.
    def rhs_builder(divergence: np.ndarray) -> np.ndarray:
        rhs = -(rho / dt) * divergence
        if state.is_fluid is not None:
            rhs = rhs * state.is_fluid  # Zero out non-fluid (solid) cells
        return rhs

    # 4. Assembly
    settings = state.config.get("solver_settings", {})
    state.ppe = {
        "A": state.operators.get("laplacian"),
        "rhs_coeff": rho / dt,
        "rhs_builder": rhs_builder,
        "solver_type": settings.get("solver_type", "PCG"),
        "tolerance": settings.get("ppe_tolerance", 1e-6),
        "max_iterations": settings.get("ppe_max_iter", 1000),
        "ppe_is_singular": is_singular
    }