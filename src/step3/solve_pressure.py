# src/step3/solve_pressure.py

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg, LinearOperator

def solve_pressure(state, rhs_ppe):
    """
    Step‑3 pressure solve using Preconditioned Conjugate Gradient (PCG).
    Optimal for Symmetric Positive Definite (SPD) Laplacian matrices.
    """
    settings = state.config.get("solver_settings", {})
    tol = settings.get("ppe_tolerance", 1e-6)
    atol = settings.get("ppe_atol", 1e-12)
    max_iter = settings.get("ppe_max_iter", 1000)
    
    A = state.ppe["A"]
    is_singular = state.ppe.get("ppe_is_singular", True)

    # 1. Create a Simple Jacobi Preconditioner (M^-1)
    # M is the diagonal of A. Solving Mz = r is just z = r / diag(A).
    diag_A = A.diagonal()
    # Avoid division by zero in solid cells
    diag_A[diag_A == 0] = 1.0 
    M_inv = diags(1.0 / diag_A)

    # 2. Flatten RHS and set Initial Guess
    b = rhs_ppe.ravel()
    x_guess = state.fields["P"].ravel()
    
    # 3. Solve using CG with the Preconditioner (M)
    # This officially makes it a 'PCG' solver.
    x, info = cg(
        A, b, 
        x0=x_guess, 
        rtol=tol, 
        atol=atol, 
        maxiter=max_iter,
        M=M_inv  # The Preconditioner
    )

    # 4. Reshape and Mean-Subtraction
    P_new = x.reshape(state.fields["P"].shape)

    if is_singular:
        mask = state.is_fluid
        if mask is not None and np.any(mask):
            P_new[mask] -= np.mean(P_new[mask])
            P_new[~mask] = 0.0
        else:
            P_new -= np.mean(P_new)

    metadata = {
        "converged": info == 0,
        "solver_status": "Success" if info == 0 else f"Failed ({info})",
        "method": "PCG (Jacobi)",
        "tolerance_used": tol,
        "is_singular": is_singular
    }

    return P_new, metadata