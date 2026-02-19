# src/step2/compute_initial_health.py

from __future__ import annotations
import numpy as np
from src.solver_state import SolverState

def compute_initial_health(state: SolverState) -> None:
    """
    Compute initial solver diagnostics using sparse operators.
    Calculates divergence, max velocity, and CFL estimates based on 
    current field values and grid constants.
    """
    # 1. Access fields (Assuming staggered grid flattening logic for sparse mult)
    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]

    # Flatten velocity components into a single vector for the sparse Divergence matrix
    # [U_flat, V_flat, W_flat]
    velocity_vector = np.concatenate([U.ravel(), V.ravel(), W.ravel()])

    # 2. Compute Divergence Norm (Sparse Matrix Multiplication)
    # Scale Guard Rule: Using @ for sparse matrix-vector product
    div_vector = state.operators["divergence"] @ velocity_vector
    div_norm = float(np.linalg.norm(div_vector))

    # 3. Compute Max Velocity Magnitude
    # Note: Components are staggered; this is a conservative estimate
    max_vel = float(np.sqrt(np.max(U**2) + np.max(V**2) + np.max(W**2)))

    # 4. Compute CFL Estimate
    dt = state.constants["dt"]
    dx = state.constants["dx"]
    dy = state.constants["dy"]
    dz = state.constants["dz"]

    # Simplified CFL for staggered grids
    cfl = float(dt * (np.max(np.abs(U))/dx + np.max(np.abs(V))/dy + np.max(np.abs(W))/dz))

    # 5. Update state.health with schema-compliant keys
    state.health = {
        "divergence_norm": div_norm,
        "max_velocity": max_vel,
        "cfl": cfl,
    }