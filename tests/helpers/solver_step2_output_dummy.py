# tests/helpers/solver_step2_output_dummy.py

import numpy as np
from scipy.sparse import csr_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Canonical dummy representing the EXACT structure of a real post‑Step‑2 SolverState.
    Updated to comply with Phase C, Rule 7 (Scale Guard) and Departmental Integrity.

    Step 2 adds:
      - Refined mask semantics (Inherited and consistent with Step 1)
      - Operators as SciPy Sparse Matrices (Laplacian, Divergence, Gradient)
      - PPE (Pressure Poisson Equation) system matrix and metadata
      - Initialized health diagnostics for Step 2 logic
    """

    # 1. Start from the refined Step 1 dummy (The Geometry/Physics foundation)
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)

    # 2. Calculate Degrees of Freedom (DOF) for matrix dimensions
    # Pressure is at cell centers (Scalar)
    dof_p = nx * ny * nz
    # Velocity components are staggered (Faces)
    dof_u = (nx + 1) * ny * nz
    dof_v = nx * (ny + 1) * nz
    dof_w = nx * ny * (nz + 1)
    total_vel_dof = dof_u + dof_v + dof_w

    # ------------------------------------------------------------------
    # 3. Mask Semantics (Refining definitions from Step 1)
    # ------------------------------------------------------------------
    # Following Step 1: 1 = fluid, 0 = boundary/solid
    state.is_fluid = (state.mask == 1)
    state.is_solid = (state.mask == 0)
    state.is_boundary_cell = np.zeros((nx, ny, nz), dtype=bool) # To be filled by Step 2 logic

    # ------------------------------------------------------------------
    # 4. Operators (Sparsity Guard Compliant)
    # Using empty CSR matrices to define the expected structure
    # ------------------------------------------------------------------
    state.operators = {
        # Laplacian matrix maps Pressure to Pressure
        "laplacian": csr_matrix((dof_p, dof_p)),
        
        # Divergence maps Velocity (U,V,W) to Pressure
        "divergence": csr_matrix((dof_p, total_vel_dof)),
        
        # Gradient maps Pressure to Velocity (U,V,W)
        "gradient": csr_matrix((total_vel_dof, dof_p)),
    }

    # ------------------------------------------------------------------
    # 5. PPE (Pressure Poisson Equation) structure
    # ------------------------------------------------------------------
    state.ppe = {
        "solver_type": "sparse_cg",  
        "A": state.operators["laplacian"],
        "tolerance": 1e-6,
        "max_iterations": 1000,
        "ppe_is_singular": True,     
        "rhs_norm": 0.0,
    }

    # ------------------------------------------------------------------
    # 6. Step‑2 Health Diagnostics
    # ------------------------------------------------------------------
    # We update the dictionary to keep existing keys or add new ones
    state.health.update({
        "divergence_norm": 0.0,
        "max_velocity": 0.0,
        "cfl": 0.0,
    })

    # 7. Progression Flag
    state.ready_for_time_loop = False 

    return state