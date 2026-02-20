# tests/step3/test_predict_velocity.py

import numpy as np
from scipy.sparse import csr_matrix
from src.step3.predict_velocity import predict_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_predict_velocity_viscous_diffusion():
    """Test if velocity increases correctly via diffusion."""
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    state.constants["dt"] = 1.0
    state.constants["mu"] = 1.0 
    state.constants["rho"] = 1.0
    
    size_u = state.fields["U"].size
    state.operators["lap_u"] = csr_matrix(np.eye(size_u))
    state.operators["advection_u"] = csr_matrix((size_u, size_u))
    state.fields["U"][:] = 1.0
    
    predict_velocity(state)
    
    # Internal velocity check: 1.0 + 1.0 * (1.0 * 1.0) = 2.0
    internal_u_star = state.intermediate_fields["U_star"]
    assert np.allclose(internal_u_star, 2.0)