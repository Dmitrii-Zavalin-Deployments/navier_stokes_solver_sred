import pytest
import numpy as np
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from src.step3.predict_velocity import predict_velocity
from src.step3.build_ppe_rhs import build_ppe_rhs

def test_full_step3_data_flow():
    """
    Ensures that the output of predict_velocity (staggered)
    is accepted by build_ppe_rhs without shape errors.
    """
    # 1. Setup
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants["rho"] = 1.0
    state.config["dt"] = 0.01
    
    # 2. Inject the staggered operators we just built
    from .staggered_mock_generator import inject_staggered_operators
    state = inject_staggered_operators(state)
    
    # 3. Predict Step (Intermediate Star Velocity)
    # Output: U_star(4,3,3), V_star(3,4,3), W_star(3,3,4)
    U_star, V_star, W_star = predict_velocity(state)
    
    # 4. RHS Step (Divergence calculation)
    # This will fail if build_ppe_rhs doesn't handle the staggered sizes
    try:
        rhs = build_ppe_rhs(state, U_star, V_star, W_star)
    except ValueError as e:
        pytest.fail(f"Integration failed due to shape mismatch: {e}")
        
    assert rhs.shape == (3, 3, 3), "RHS should be cell-centered"