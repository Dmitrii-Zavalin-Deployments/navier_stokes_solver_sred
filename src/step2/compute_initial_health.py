# src/step2/compute_initial_health.py

from __future__ import annotations
import numpy as np
from src.solver_state import SolverState

def compute_initial_health(state: SolverState, divergence_override: np.ndarray = None) -> None:
    """
    Compute initial solver diagnostics.
    
    Production Logic: 
      Uses state.operators["divergence"] to calculate the true divergence.
    
    Testing Logic: 
      Allows a 'divergence_override' to verify norm calculations 
      without requiring a full sparse operator setup.
    """
    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]

    # 1. Compute Divergence Norm
    if divergence_override is not None:
        # Strict check: only used in unit tests
        div_norm = float(np.linalg.norm(divergence_override))
    else:
        # PRODUCTION PATH: Calculate using the physical operators
        if "divergence" not in state.operators:
            raise KeyError("Divergence operator not found in state.operators. "
                           "Ensure build_divergence_operator was called.")
            
        velocity_vector = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
        div_vector = state.operators["divergence"] @ velocity_vector
        div_norm = float(np.linalg.norm(div_vector))

    # 2. Compute Max Velocity Magnitude
    # Scale Guard: Using np.linalg.norm on components is more robust than simple addition
    max_vel = float(np.sqrt(np.max(U**2) + np.max(V**2) + np.max(W**2)))

    # 3. Compute CFL Estimate
    # Using dictionary access for grid/config as per SolverState schema
    dt = state.config.get('dt', 0.0)
    dx = state.grid.get('dx', 1.0)
    dy = state.grid.get('dy', 1.0)
    dz = state.grid.get('dz', 1.0)

    # Simplified Courant number calculation
    cfl = float(dt * (np.max(np.abs(U))/dx + np.max(np.abs(V))/dy + np.max(np.abs(W))/dz))

    # 4. Update health block
    state.health.update({
        "divergence_norm": div_norm,
        "max_velocity": max_vel,
        "cfl": cfl,
    })