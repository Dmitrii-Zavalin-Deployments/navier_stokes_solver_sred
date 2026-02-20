# src/step3/solve_pressure.py

import numpy as np
from scipy.sparse.linalg import cg

def solve_pressure(state, rhs_ppe):
    """
    Step‑3 pressure solve.
    Solves ∇² p = rhs using SciPy Sparse iterative solvers (Conjugate Gradient).
    """
    # 1. Extract Numerical Settings from config
    settings = state.config.get("solver_settings", {})
    tol = settings.get("ppe_tolerance", 1e-6)
    max_iter = settings.get("ppe_max_iter", 1000)
    
    # 2. Extract Operator and Singularity metadata
    A = state.ppe["A"]
    is_singular = state.ppe.get("ppe_is_singular", True)

    # 3. Flatten RHS for the linear solver
    b = rhs_ppe.ravel()
    
    # 4. Solve using Conjugate Gradient
    # FIX: Use 'rtol' instead of 'tol' for SciPy 1.11+ compatibility
    # We also provide atol=0 to let rtol do the heavy lifting
    x, info = cg(A, b, rtol=tol, atol=0, maxiter=max_iter)

    # 5. Reshape back to the 3D Pressure field shape
    P_new = x.reshape(state.fields["P"].shape)

    # 6. Handle Singularity (Fix the Pressure constant)
    if is_singular:
        mask = state.is_fluid
        if mask is not None and np.any(mask):
            P_new[mask] -= np.mean(P_new[mask])
        else:
            P_new -= np.mean(P_new)

    # 7. Metadata
    metadata = {
        "converged": info == 0,
        "solver_status": info,
        "tolerance_used": tol
    }

    return P_new, metadata