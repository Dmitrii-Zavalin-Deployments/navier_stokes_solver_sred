import numpy as np
from scipy.sparse import csr_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 2 Dummy: Mimics the exact output of orchestrate_step2.
    
    Constitutional Role: 
    Inherits Step 1 data and populates the 'Mathematical' safes:
    - Operators (Sparse Matrices aligned to Staggered Grid DOFs)
    - PPE System (A and Preconditioner)
    - Initial Health (Vitals)
    - Advection Stencils
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Precise staggered grid DOF calculation
    dof_p = nx * ny * nz              # 4*4*4 = 64
    dof_u = (nx + 1) * ny * nz        # 5*4*4 = 80
    dof_v = nx * (ny + 1) * nz        # 4*5*4 = 80
    dof_w = nx * ny * (nz + 1)        # 4*4*5 = 80
    total_vel_dof = dof_u + dof_v + dof_w

    # 1. Hydrate the Gradient/Divergence Operators
    state.operators._divergence = csr_matrix((dof_p, total_vel_dof))
    state.operators._grad_x = csr_matrix((dof_u, dof_p))
    state.operators._grad_y = csr_matrix((dof_v, dof_p))
    state.operators._grad_z = csr_matrix((dof_w, dof_p))
    
    # 2. Laplacian Alignment:
    # Must match velocity DOF (80) to satisfy the Predictor (U_star = U + dt * (laplacian @ U))
    state.operators._laplacian = csr_matrix((dof_u, dof_u))
    
    # 3. PPE Alignment:
    # Must match pressure DOF (64) to satisfy the Pressure Poisson Equation (A @ p = div)
    state.ppe._A = csr_matrix((dof_p, dof_p))
    
    # 4. Advection & Health
    state.advection._weights = np.zeros((total_vel_dof, 8))
    state.advection._indices = np.zeros((total_vel_dof, 8), dtype=int)
    state.health.is_stable = True

    return state
