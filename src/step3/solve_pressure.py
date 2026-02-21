# src/step3/solve_pressure.py

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg

def solve_pressure(state, rhs_ppe):
    settings = state.config.get("solver_settings", {})
    tol = settings.get("ppe_tolerance", 1e-6)
    atol = settings.get("ppe_atol", 1e-12)
    max_iter = settings.get("ppe_max_iter", 1000)
    
    A = state.ppe["A"]
    is_singular = state.ppe.get("ppe_is_singular", True)

    diag_A = A.diagonal()
    safe_diag = np.where(np.abs(diag_A) < 1e-12, 1.0, diag_A)
    M_inv = diags(1.0 / safe_diag)

    b = rhs_ppe.ravel()
    x_guess = state.fields["P"].ravel()
    
    x, info = cg(A, b, x0=x_guess, rtol=tol, atol=atol, maxiter=max_iter, M=M_inv)

    # RECOVERY: If CG fails or returns NaNs due to singularity
    if np.any(np.isnan(x)):
        x = np.zeros_like(b) # Fallback to zero if exploded

    P_new = x.reshape(state.fields["P"].shape)

    if is_singular:
        mask = state.is_fluid
        if mask is not None and np.any(mask):
            fluid_mean = np.mean(P_new[mask])
            P_new[mask] -= fluid_mean
            P_new[~mask] = 0.0
        else:
            P_new -= np.mean(P_new)

    metadata = {
        "converged": info == 0 and not np.any(np.isnan(x)),
        "solver_status": "Success" if (info == 0 and not np.any(np.isnan(x))) else "Singular/Failed",
        "method": "PCG (Jacobi)",
        "is_singular": is_singular
    }

    return P_new, metadata