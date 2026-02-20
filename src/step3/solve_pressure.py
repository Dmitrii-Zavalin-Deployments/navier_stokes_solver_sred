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
    atol = settings.get("ppe_atol", 1e-12)  # Fetched from config with safe fallback
    max_iter = settings.get("ppe_max_iter", 1000)
    
    # 2. Extract Operator and Singularity metadata
    A = state.ppe["A"]
    is_singular = state.ppe.get("ppe_is_singular", True)

    # 3. Flatten RHS and set Initial Guess
    b = rhs_ppe.ravel()
    x_guess = state.fields["P"].ravel()
    
    # 4. Solve using Conjugate Gradient (CG)
    # Using both rtol and atol for robust termination criteria
    x, info = cg(
        A, b, 
        x0=x_guess, 
        rtol=tol, 
        atol=atol, 
        maxiter=max_iter
    )

    # 5. Reshape and Mean-Subtraction (Singularity Fix)
    P_new = x.reshape(state.fields["P"].shape)

    if is_singular:
        mask = state.is_fluid
        if mask is not None and np.any(mask):
            fluid_mean = np.mean(P_new[mask])
            P_new[mask] -= fluid_mean
            P_new[~mask] = 0.0
        else:
            P_new -= np.mean(P_new)

    # 6. Metadata
    metadata = {
        "converged": info == 0,
        "solver_status": "Success" if info == 0 else f"Failed (info={info})",
        "tolerance_used": tol,
        "absolute_tolerance_used": atol
    }

    return P_new, metadata