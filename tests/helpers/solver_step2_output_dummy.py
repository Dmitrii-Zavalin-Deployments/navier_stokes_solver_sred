# tests/helpers/solver_step2_output_dummy.py

import numpy as np
from scipy.sparse import csr_matrix
from src.solver_state import SolverState
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Updated to comply with Phase C, Rule 7 (Scale Guard).
    Operators are now SciPy Sparse Matrices, not lambdas.
    """
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)

    # Calculate total DOF (Degrees of Freedom) for the grid
    # For a pressure-centered grid (nx*ny*nz)
    dof_p = nx * ny * nz
    
    # ------------------------------------------------------------------
    # Operators (Sparsity Guard Compliant)
    # We use identity matrices as placeholders for structure validation
    # ------------------------------------------------------------------
    state.operators = {
        "laplacian": csr_matrix((dof_p, dof_p)), # The A matrix for PPE
        "div": None,    # To be built as sparse matrices
        "grad": None,   # To be built as sparse matrices
    }

    # ------------------------------------------------------------------
    # PPE metadata
    # ------------------------------------------------------------------
    state.ppe = {
        "solver_type": "sparse_cg", # Use a real sparse solver name
        "A": csr_matrix((dof_p, dof_p)), 
        "tolerance": 1e-6,
    }

    state.health = {
        "divergence_norm": 0.0,
        "max_velocity": 0.0,
        "cfl": 0.0,
    }

    return state