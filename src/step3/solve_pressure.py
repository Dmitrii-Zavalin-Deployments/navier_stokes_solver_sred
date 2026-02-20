# src/step3/solve_pressure.py

import numpy as np
from scipy.sparse.linalg import cg

def solve_pressure(state, rhs_ppe):
    """
    Step‑3 pressure solve.
    Solves ∇² p = rhs using SciPy Sparse iterative solvers (Conjugate Gradient).
    
    Adheres to Rule 7 (Scale Guard) and Zero-Debt Mandate:
    - Strictly maps to 'solver_settings' in state.config.
    - Uses sparse matrix A from state.ppe.
    - Performs physical mean-subtraction for singular systems.
    """
    # 1. Extract Numerical Settings from config
    settings = state.config.get("solver_settings", {})
    tol = settings.get("ppe_tolerance", 1e-6)
    max_iter = settings.get("ppe_max_iter", 1000)
    
    # 2. Extract Operator and Singularity metadata
    A = state.ppe["A"]
    is_singular = state.ppe.get("ppe_is_singular", True)

    # 3. Flatten RHS for the linear solver
    # Linear solvers expect a 1D vector: Ax = b
    b = rhs_ppe.ravel()
    
    # 4. Solve using Conjugate Gradient (optimized for Symmetric Positive Definite matrices)
    # x: pressure vector, info: convergence status (0=success, >0=no convergence)
    x, info = cg(A, b, tol=tol, maxiter=max_iter)

    # 5. Reshape back to the 3D Pressure field shape
    P_new = x.reshape(state.fields["P"].shape)

    # 6. Handle Singularity (Fix the Pressure constant)
    # In pure Neumann systems, pressure is relative; we fix the mean to 0.0.
    if is_singular:
        mask = state.is_fluid
        if mask is not None and np.any(mask):
            # Only average over actual fluid cells for physical accuracy
            P_new[mask] -= np.mean(P_new[mask])
        else:
            P_new -= np.mean(P_new)

    # 7. Metadata for diagnostics and Step 3 orchestration
    metadata = {
        "converged": info == 0,
        "solver_status": info,
        "tolerance_used": tol
    }

    return P_new, metadata