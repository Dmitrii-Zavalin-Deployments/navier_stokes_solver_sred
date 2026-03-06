# src/step2/diffusion.py

from __future__ import annotations

from src.solver_state import SolverState
from src.step2.operators import build_laplacian


def build_diffusion_operators(state: SolverState) -> None:
    """
    Step 2 Logic: Build the Diffusion operator (μ∇²v).
    
    Rule 8 Compliance: Consumes central-difference operators 
    from the operator library to maintain SSoT.
    """
    # 1. Retrieve the pre-built Gradient operator
    # The Laplacian is constructed from the divergence and gradient pair
    Gx = state.operators.grad_x
    D = state.operators.divergence
    
    # 2. Build the Laplacian (L = D @ G)
    # This represents the second-order central difference for diffusion
    L = build_laplacian(Gx, D)
    
    # 3. Commit to state
    # We multiply by viscosity (mu) later in the predictor step
    state.diffusion.laplacian = L
    
    # Deterministic Check
    assert state.diffusion.laplacian.shape == (state.grid.nx * state.grid.ny * state.grid.nz, 
                                               state.grid.nx * state.grid.ny * state.grid.nz)