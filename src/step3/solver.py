# src/step3/solver.py

import numpy as np
from scipy.sparse.linalg import cg
from src.solver_state import SolverState

def solve_pressure(state: SolverState) -> str:
    """
    Step 3.2: Pressure Poisson Solve.
    Enforces 'F' order to maintain staggered grid alignment.
    Rule 5 Compliance: No defaults. Accesses config properties directly.
    """
    rho = state.density
    dt = state.dt
    
    # 1. Build RHS: b = (rho/dt) * Divergence(V_star)
    # Concatenate using Fortran order to match grid topology
    v_star_flat = np.concatenate([
        state.fields.U_star.flatten(order='F'), 
        state.fields.V_star.flatten(order='F'), 
        state.fields.W_star.flatten(order='F')
    ])
    
    # Calculate divergence and reshape correctly
    div_v_star = (state.operators.divergence @ v_star_flat).reshape(state.fields.P.shape, order='F')
    rhs = (rho / dt) * div_v_star

    # 2. Linear Solve: AP = b
    # Direct property access triggers _get_safe validation. 
    # If these are None, the solver will raise a RuntimeError.
    p_flat, info = cg(
        state.ppe._A, 
        rhs.flatten(order='F'), 
        x0=state.fields.P.flatten(order='F'),
        rtol=state.config.ppe_tolerance,
        atol=state.config.ppe_atol,
        maxiter=state.config.ppe_max_iter
    )
    
    # Update pressure field in-place with correct memory layout
    state.fields.P = p_flat.reshape(state.fields.P.shape, order='F')
    
    return "converged" if info == 0 else "failed"