# src/step2/operators.py

import scipy.sparse as sp
from src.solver_state import SolverState

def build_numerical_operators(state: SolverState) -> None:
    """
    Step 2 Logic: Calculate DOF and prepare Sparse Matrix shells.
    Rule 1: Scale Guard. Operators remain in scipy.sparse format.
    """
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    dof_p = nx * ny * nz
    dof_u = (nx + 1) * ny * nz
    dof_v = nx * (ny + 1) * nz
    dof_w = nx * ny * (nz + 1)
    total_vel_dof = dof_u + dof_v + dof_w

    # Initialize Sparse Containers (Contract-compliant shapes)
    state.operators.divergence = sp.csr_matrix((dof_p, total_vel_dof))
    state.operators.grad_x = sp.csr_matrix((dof_u, dof_p))
    state.operators.grad_y = sp.csr_matrix((dof_v, dof_p))
    state.operators.grad_z = sp.csr_matrix((dof_w, dof_p))
    state.operators.laplacian = sp.csr_matrix((dof_p, dof_p))