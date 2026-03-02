import numpy as np
from scipy.sparse import csr_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 2 Dummy: Aligned to the Frozen Physics Schema.
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # DOFs for 4x4x4 grid
    dof_p = nx * ny * nz              # 64
    dof_u = (nx + 1) * ny * nz        # 80
    dof_v = nx * (ny + 1) * nz        # 80
    dof_w = nx * ny * (nz + 1)        # 80
    total_vel_dof = dof_u + dof_v + dof_w

    # Standard Operators
    state.operators._divergence = csr_matrix((dof_p, total_vel_dof))
    state.operators._grad_x = csr_matrix((dof_u, dof_p))
    state.operators._grad_y = csr_matrix((dof_v, dof_p))
    state.operators._grad_z = csr_matrix((dof_w, dof_p))
    
    # Frozen Reality: Step 2 produces a Laplacian matching the pressure grid
    state.operators._laplacian = csr_matrix((dof_p, dof_p))
    
    # Advection Attributes (Required by Step 3 Predictor)
    # If these aren't in OperatorStorage, we attach them to the instance
    state.operators.advection_u = csr_matrix((dof_u, dof_u))
    state.operators.advection_v = csr_matrix((dof_v, dof_v))
    state.operators.advection_w = csr_matrix((dof_w, dof_w))

    state.ppe._A = state.operators._laplacian
    state.advection._weights = np.zeros((total_vel_dof, 8))
    state.advection._indices = np.zeros((total_vel_dof, 8), dtype=int)
    state.health.is_stable = True

    return state
