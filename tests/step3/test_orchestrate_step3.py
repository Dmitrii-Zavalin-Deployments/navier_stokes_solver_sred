# tests/step3/test_orchestrate_step3.py

import numpy as np
from scipy.sparse import csr_matrix, eye
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def _prepare_mock_state(nx):
    state = make_step2_output_dummy(nx=nx, ny=nx, nz=nx)
    state.operators["grad_x"] = lambda P: np.zeros_like(state.fields["U"])
    state.operators["grad_y"] = lambda P: np.zeros_like(state.fields["V"])
    state.operators["grad_z"] = lambda P: np.zeros_like(state.fields["W"])
    
    size_p = nx**3
    total_vel = state.fields["U"].size + state.fields["V"].size + state.fields["W"].size
    state.operators["divergence"] = csr_matrix((size_p, total_vel))
    state.ppe["A"] = eye(size_p, format="csr")
    state.ppe["ppe_is_singular"] = False
    
    # Add missing keys for predict_velocity
    state.operators["lap_u"] = csr_matrix((state.fields["U"].size, state.fields["U"].size))
    state.operators["advection_u"] = csr_matrix((state.fields["U"].size, state.fields["U"].size))
    # ... and for V, W if necessary ...

    state.config["solver_settings"] = {"ppe_tolerance": 1e-6, "ppe_max_iter": 10}
    return state