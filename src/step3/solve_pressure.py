# src/step3/solve_pressure.py

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg

def solve_pressure(state, rhs_ppe):
    """
    Step‑3 pressure solve using Preconditioned Conjugate Gradient (PCG).
    Optimal for Symmetric Positive Definite (SPD) Laplacian matrices.
    """
    # 1. Extract Numerical Settings from config
    settings = state.config.get("solver_settings", {})
    tol = settings.get("ppe_tolerance", 1e-6)
    atol = settings.get("ppe_atol", 1e-12)
    max_iter = settings.get("ppe_max_iter", 1000)
    
    # 2. Extract Operator and Singularity metadata
    A = state.ppe["A"]
    is_singular = state.ppe.get("ppe_is_singular", True)

    # 3. Create a Simple Jacobi Preconditioner (M^-1)
    # Scales the system by the inverse of the diagonal to improve conditioning.
    diag_A = A.diagonal()
    diag_A[diag_A == 0] = 1.0  # Avoid division by zero in solid/masked cells
    M_inv = diags(1.0 / diag_A)

    # 4. Flatten RHS and set Initial Guess
    b = rhs_ppe.ravel()
    x_guess = state.fields["P"].ravel()
    
    # 5. Solve using CG with the Preconditioner (M)
    x, info = cg(
        A, b, 
        x0=x_guess, 
        rtol=tol, 
        atol=atol, 
        maxiter=max_iter,
        M=M_inv
    )

    # 6. Reshape and Mean-Subtraction (Singularity Fix)
    P_new = x.reshape(state.fields["P"].shape)

    if is_singular:
        mask = state.is_fluid
        if mask is not None and np.any(mask):
            fluid_mean = np.mean(P_new[mask])
            P_new[mask] -= fluid_mean
            P_new[~mask] = 0.0
        else:
            P_new -= np.mean(P_new)

    # 7. Metadata for history (Synchronized with test expectations)
    metadata = {
        "converged": info == 0,
        "solver_status": "Success" if info == 0 else f"Failed ({info})",
        "method": "PCG (Jacobi)",
        "tolerance_used": tol,
        "absolute_tolerance_used": atol,  # Required by test suite
        "is_singular": is_singular
    }

    return P_new, metadata