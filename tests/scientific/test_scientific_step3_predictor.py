# tests/scientific/test_scientific_step3_predictor.py

import pytest
import numpy as np
from scipy import sparse
from src.solver_state import SolverState

@pytest.fixture
def state_predictor():
    """Fixture to set up a valid state for predictor physics."""
    state = SolverState()
    # 1. Setup Physics Constants
    # Formula: nu = mu / rho (0.001 / 1000 = 1e-6)
    state.config._fluid_properties = {"density": 1000.0, "viscosity": 1.0} # nu = 0.001
    state.config._simulation_parameters = {"time_step": 0.1, "total_time": 1.0, "output_interval": 1}
    state.config._external_forces = {"force_vector": [1.0, 0.0, 0.0]} # Force in X
    
    # 2. Allocate Fields (3x3x3 grid)
    # U: (4,3,3) = 36 elements
    state.fields._U = np.ones((4, 3, 3)) 
    state.fields._V = np.zeros((3, 4, 3))
    state.fields._W = np.zeros((3, 3, 4))
    
    # 3. Setup Operators (Identity for Laplacian to test flow)
    # grad_x/y/z and laplacian are accessed by Predictor
    state.operators._laplacian = sparse.eye(36, 36)
    
    return state

## =========================================================
## PHYSICS & FORMULA VERIFICATION
## =========================================================

def test_predict_velocity_formula_u(state_predictor, capsys):
    """
    Verify U* = U + dt * (nu * Lap(U) + force_u)
    Initial U = 1.0, dt = 0.1, nu = 0.001, Lap(U) = 1.0, force_u = 1.0
    Expected: 1.0 + 0.1 * (0.001 * 1.0 + 1.0) = 1.1001
    """
    from src.step3.predictor import predict_velocity
    
    predict_velocity(state_predictor)
    
    expected_u_star = 1.1001
    assert np.allclose(state_predictor.fields.U_star, expected_u_star)
    
    # Verify Debug Coverage
    captured = capsys.readouterr()
    assert "Predicting U_star" in captured.out
    assert "Nu=1.000000e-03" in captured.out

def test_predict_velocity_missing_operator(state_predictor):
    """
    Logic Gate: Ensure RuntimeError is raised if Laplacian is missing.
    """
    from src.step3.predictor import predict_velocity
    state_predictor.operators._laplacian = None
    
    with pytest.raises(RuntimeError, match="Operator 'laplacian' is missing"):
        predict_velocity(state_predictor)

def test_predict_velocity_nan_detection(state_predictor, capsys):
    """
    Debug Check: Ensure the 'Critical NaN' message prints if physics explode.
    """
    from src.step3.predictor import predict_velocity
    # Inject NaN into the source field
    state_predictor.fields.U[0,0,0] = np.nan
    
    predict_velocity(state_predictor)
    
    captured = capsys.readouterr()
    assert "CRITICAL: NaN detected" in captured.out

def test_predict_velocity_v_w_components(state_predictor):
    """
    Verify V and W components are updated using their respective force vectors.
    """
    from src.step3.predictor import predict_velocity
    # Set a force in Y and Z
    state_predictor.config._external_forces = {"force_vector": [0.0, 5.0, 10.0]}
    
    # V and W operators need to be sized for their specific staggered grid points
    # V: (3,4,3) = 36, W: (3,3,4) = 36. Luckily both are 36 for 3x3x3.
    state_predictor.operators._laplacian = sparse.eye(36, 36)
    
    predict_velocity(state_predictor)
    
    # V_star = 0 + 0.1 * (0.001 * 0 + 5.0) = 0.5
    assert np.allclose(state_predictor.fields.V_star, 0.5)
    # W_star = 0 + 0.1 * (0.001 * 0 + 10.0) = 1.0
    assert np.allclose(state_predictor.fields.W_star, 1.0)
