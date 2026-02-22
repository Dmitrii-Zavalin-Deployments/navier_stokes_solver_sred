# tests/helpers/solver_step2_output_dummy.py

import numpy as np
from scipy.sparse import csr_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Canonical dummy representing the EXACT structure of a real post‑Step‑2 SolverState.
    Updated to comply with Phase C, Rule 7 (Scale Guard) and Departmental Integrity.

    Step 2 adds:
      - Refined mask semantics (Inherited from Step 1, cast to JSON-safe lists)
      - Operators as SciPy Sparse Matrices (Laplacian, Divergence, Gradient)
      - PPE (Pressure Poisson Equation) system matrix and metadata
      - Initialized health diagnostics for Step 2 logic
    """

    # 1. Start from the refined Step 1 dummy (The Geometry/Physics foundation)
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)

    # 2. Calculate Degrees of Freedom (DOF) for matrix dimensions
    dof_p = nx * ny * nz
    dof_u = (nx + 1) * ny * nz
    dof_v = nx * (ny + 1) * nz
    dof_w = nx * ny * (nz + 1)
    total_vel_dof = dof_u + dof_v + dof_w

    # ------------------------------------------------------------------
    # 3. Mask Semantics (Fixing the Boolean Leak)
    # ------------------------------------------------------------------
    # Convert inherited mask back to numpy for element-wise comparison
    mask_arr = np.array(state.mask)
    
    # Cast boolean results to int then to list to satisfy JSON 'array' contract
    state.is_fluid = (mask_arr == 1).astype(int).tolist()
    state.is_solid = (mask_arr == 0).astype(int).tolist()
    state.is_boundary_cell = np.zeros((nx, ny, nz), dtype=int).tolist() 

    # ------------------------------------------------------------------
    # 4. Operators (Sparsity Guard Compliant)
    # ------------------------------------------------------------------
    state.operators = {
        "laplacian": csr_matrix((dof_p, dof_p)),
        "divergence": csr_matrix((dof_p, total_vel_dof)),
        "gradient": csr_matrix((total_vel_dof, dof_p)),
    }

    # ------------------------------------------------------------------
    # 5. PPE (Pressure Poisson Equation) structure
    # ------------------------------------------------------------------
    state.ppe = {
        "dimension": dof_p,
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
    state.health.update({
        "divergence_norm": 0.0,
        "max_velocity": 0.0,
        "cfl": 0.0,
    })

    # 7. Progression Flag
    state.ready_for_time_loop = False 

    return state