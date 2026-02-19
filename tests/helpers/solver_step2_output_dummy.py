# tests/helpers/solver_step2_output_dummy.py

import numpy as np
from scipy.sparse import csr_matrix
from src.solver_state import SolverState
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Canonical dummy representing the EXACT structure of a real post‑Step‑2 SolverState.
    Updated to comply with Phase C, Rule 7 (Scale Guard).

    Step 2 adds:
      - Refined mask semantics (is_fluid, is_boundary_cell, is_solid)
      - Operators as SciPy Sparse Matrices (Laplacian, Divergence, Gradient)
      - PPE (Pressure Poisson Equation) system matrix and metadata
      - Initialized health diagnostics for Step 2 logic
    """

    # Start from the frozen Step 1 dummy
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)

    # 1. Calculate Degrees of Freedom (DOF) for matrix dimensions
    # Pressure is at cell centers
    dof_p = nx * ny * nz
    # Velocity components are staggered (Simplified DOF for dummy structure)
    dof_u = (nx + 1) * ny * nz
    dof_v = nx * (ny + 1) * nz
    dof_w = nx * ny * (nz + 1)
    total_vel_dof = dof_u + dof_v + dof_w

    # ------------------------------------------------------------------
    # 2. Mask Semantics (Step 2 refines these for matrix indexing)
    # ------------------------------------------------------------------
    state.is_fluid = state.mask == 0
    state.is_boundary_cell = (state.mask == 1) | (state.mask == -1)
    state.is_solid = state.mask == -1 # Example distinction

    # ------------------------------------------------------------------
    # 3. Operators (Sparsity Guard Compliant)
    # Using empty CSR matrices to define the expected structure
    # ------------------------------------------------------------------
    state.operators = {
        # Laplacian matrix (A) maps P -> P: (dof_p, dof_p)
        "laplacian": csr_matrix((dof_p, dof_p)),
        
        # Divergence maps U,V,W -> P: (dof_p, total_vel_dof)
        "divergence": csr_matrix((dof_p, total_vel_dof)),
        
        # Gradient maps P -> U,V,W: (total_vel_dof, dof_p)
        "gradient": csr_matrix((total_vel_dof, dof_p)),
    }

    # ------------------------------------------------------------------
    # 4. PPE (Pressure Poisson Equation) structure
    # ------------------------------------------------------------------
    state.ppe = {
        "solver_type": "sparse_cg",  # Conjugate Gradient for sparse matrices
        "A": state.operators["laplacian"],
        "tolerance": 1e-6,
        "max_iterations": 1000,
        "ppe_is_singular": True,     # Often true for pure Neumann systems
        "rhs_norm": 0.0,
    }

    # ------------------------------------------------------------------
    # 5. Step‑2 Health Diagnostics
    # ------------------------------------------------------------------
    state.health = {
        "divergence_norm": 0.0,
        "max_velocity": 0.0,
        "cfl": 0.0,
    }

    # 6. Flag the state progression
    state.ready_for_time_loop = False 

    return state