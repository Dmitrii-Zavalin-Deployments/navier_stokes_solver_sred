# tests/helpers/solver_step2_output_dummy.py

import numpy as np
from scipy.sparse import csr_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def make_step2_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 2 Dummy: Mimics the exact output of orchestrate_step2.
    
    Constitutional Role: 
    Inherits Step 1 data and populates the 'Mathematical' safes:
    - Operators (Sparse Matrices)
    - PPE System (A and Preconditioner)
    - Initial Health (Vitals)
    - Advection Stencils
    """
    # 1. Start from the validated Step 1 foundation
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)

    # 2. Degrees of Freedom (For matrix sizing)
    dof_p = nx * ny * nz
    # Velocity components on a staggered grid
    dof_u = (nx + 1) * ny * nz
    dof_v = nx * (ny + 1) * nz
    dof_w = nx * ny * (nz + 1)
    total_vel_dof = dof_u + dof_v + dof_w

    # ------------------------------------------------------------------
    # 3. Numerical Config (from load_numerical_config)
    # ------------------------------------------------------------------
    state.config.ppe_atol = 1e-12
    state.config.ppe_max_iter = 1000

    # ------------------------------------------------------------------
    # 4. Sparse Operators (from build_operators)
    # ------------------------------------------------------------------
    # Note: We use the property setters in state.operators
    state.operators.divergence = csr_matrix((dof_p, total_vel_dof))
    state.operators.grad_x = csr_matrix((dof_u, dof_p))
    state.operators.grad_y = csr_matrix((dof_v, dof_p))
    state.operators.grad_z = csr_matrix((dof_w, dof_p))
    state.operators.laplacian = csr_matrix((dof_p, dof_p))

    # ------------------------------------------------------------------
    # 5. Advection & PPE Structures (from prepare_ppe_structure)
    # ------------------------------------------------------------------
    state.advection.weights = np.zeros((total_vel_dof, 8))
    state.advection.indices = np.zeros((total_vel_dof, 8), dtype=int)
    
    # The PPE system matrix is typically the Laplacian operator
    state.ppe.A = state.operators.laplacian
    state.ppe.preconditioner = None  # Initialized as None or identity

    # ------------------------------------------------------------------
    # 6. Initialization Health (from compute_initial_health)
    # ------------------------------------------------------------------
    state.health.max_u = 0.0
    state.health.divergence_norm = 0.0
    state.health.is_stable = True
    state.health.post_correction_divergence_norm = 0.0

    # ------------------------------------------------------------------
    # 7. Progression State
    # ------------------------------------------------------------------
    # Step 2 is mathematical readiness, but Step 4 (BCs) usually flips this bit.
    state.ready_for_time_loop = False 

    return state