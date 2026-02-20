# tests/step3/test_ppe_rhs_shape.py

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.predict_velocity import predict_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def _wire_mock_operators(state):
    """
    Wire sparse matrices for prediction and divergence to avoid KeyErrors.
    """
    nx, ny, nz = state.grid["nx"], state.grid["ny"], state.grid["nz"]
    
    # DOF counts
    size_u = (nx + 1) * ny * nz
    size_v = nx * (ny + 1) * nz
    size_w = nx * ny * (nz + 1)
    size_p = nx * ny * nz
    total_vel_dof = size_u + size_v + size_w

    # Prediction operators (Laplacian & Advection)
    state.operators["lap_u"] = csr_matrix((size_u, size_u))
    state.operators["lap_v"] = csr_matrix((size_v, size_v))
    state.operators["lap_w"] = csr_matrix((size_w, size_w))
    
    state.operators["advection_u"] = csr_matrix((size_u, size_u))
    state.operators["advection_v"] = csr_matrix((size_v, size_v))
    state.operators["advection_w"] = csr_matrix((size_w, size_w))

    # Divergence operator [P_size x Total_Vel_Size]
    state.operators["divergence"] = csr_matrix((size_p, total_vel_dof))

def test_ppe_rhs_shape():
    """
    RHS of PPE must have the same shape as the pressure field.
    """
    # 1. Setup State
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants["dt"] = 0.1
    state.constants["rho"] = 1.0
    _wire_mock_operators(state)

    # 2. Predict velocity (Updates state.intermediate_fields)
    predict_velocity(state)
    
    U_star = state.intermediate_fields["U_star"]
    V_star = state.intermediate_fields["V_star"]
    W_star = state.intermediate_fields["W_star"]

    # 3. Build RHS
    rhs = build_ppe_rhs(state, U_star, V_star, W_star)

    # 4. Assertions
    assert rhs.shape == state.fields["P"].shape
    assert isinstance(rhs, np.ndarray)