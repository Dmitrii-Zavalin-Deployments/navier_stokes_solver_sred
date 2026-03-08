# src/step3/ppe_solver.py

import numpy as np
from .ppe import compute_ppe_rhs
from .ops.sor_stencil import compute_sor_stencil
from .core.grid_utils import get_interior_slices

def solve_pressure_poisson(state):
    """
    Solves \nabla^2 p^{n+1} = RHS using Successive Over-Relaxation (SOR).
    """
    # 1. Hydrate configuration parameters
    cfg = state.config.solver_settings
    omega = cfg.get("ppe_omega", 1.5)
    max_iter = cfg.get("ppe_max_iter", 1000)
    tol = cfg.get("ppe_tolerance", 1e-6)
    
    # 2. Setup grid and RHS
    dx, dy, dz = state.grid.dx, state.grid.dy, state.grid.dz
    dx2, dy2, dz2 = dx**2, dy**2, dz**2
    stencil_denom = 2.0 * (1/dx2 + 1/dy2 + 1/dz2)
    
    rhs = compute_ppe_rhs(
        state.fields.v_star, 
        state.fields.P, 
        dx, dy, dz, 
        state.config.fluid_properties["density"], 
        state.config.simulation_parameters["time_step"]
    )
    
    # 3. Initialize pressure field and interior slice mask
    p = state.fields.P.copy()
    interior = get_interior_slices()
    
    # 4. SOR Iteration Loop
    for _ in range(max_iter):
        p_old = p.copy()
        
        # Calculate Laplacian stencil residual using the modular operator
        stencil_val = compute_sor_stencil(p, dx2, dy2, dz2, stencil_denom, rhs)
        
        # Apply the SOR update rule using the interior slice utility
        p[interior] = (1 - omega) * p[interior] + \
                      (omega / stencil_denom) * stencil_val
        
        # Convergence Check: L2 norm of the change
        if np.linalg.norm(p - p_old) < tol:
            break
            
    return p