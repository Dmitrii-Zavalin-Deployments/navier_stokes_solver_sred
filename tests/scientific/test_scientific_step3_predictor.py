# tests/scientific/test_scientific_step3_predictor.py

import pytest
import numpy as np
from scipy import sparse
from src.solver_state import SolverState

class AttributeDict(dict):
    """A dict that allows dot notation access, satisfying both type-checks and solver logic."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

@pytest.fixture
def state_predictor():
    """
    Fixture to set up a valid state for full predictor physics.
    Uses AttributeDict to pass the 'isinstance(dict)' check while supporting dot access.
    """
    state = SolverState()
    
    # 1. Physics Setup - Satisfies isinstance(value, dict) in base_container.py
    state.config.fluid_properties = AttributeDict({
        "density": 1000.0,
        "viscosity": 1.0
    })

    state.config.simulation_parameters = AttributeDict({
        "time_step": 0.1
    })

    state.config.external_forces = AttributeDict({
        "force_vector": [1.0, 0.0, 0.0]
    })
    
    # 2. Field Allocation (3x3x3 grid)
    state.fields._U = np.ones((4, 3, 3)) 
    state.fields._V = np.zeros((3, 4, 3))
    state.fields._W = np.zeros((3, 3, 4))
    
    # 3. Operator Mocking
    eye36 = sparse.eye(36, 36)
    state.operators._laplacian = eye36
    state.operators._advection_u = eye36
    state.operators._advection_v = eye36
    state.operators._advection_w = eye36
    
    return state

## =========================================================
## PHYSICS & FORMULA VERIFICATION
## =========================================================

def test_predict_velocity_full_physics(state_predictor):
    from src.step3.predictor import predict_velocity
    predict_velocity(state_predictor)
    assert np.allclose(state_predictor.fields.U_star, 1.0001)

def test_predict_velocity_missing_operator(state_predictor):
    from src.step3.predictor import predict_velocity
    state_predictor.operators._advection_u = None
    with pytest.raises(RuntimeError):
        predict_velocity(state_predictor)

def test_predict_velocity_instability_debug(state_predictor, capsys):
    from src.step3.predictor import predict_velocity
    state_predictor.fields.U[0,0,0] = np.inf
    predict_velocity(state_predictor)
    captured = capsys.readouterr()
    assert "CRITICAL: Predictor Instability" in captured.out

def test_predict_velocity_component_isolation(state_predictor):
    from src.step3.predictor import predict_velocity
    state_predictor.config.external_forces.force_vector = [1.0, 2.0, 3.0]
    predict_velocity(state_predictor)
    assert np.allclose(state_predictor.fields.U_star, 1.0001)
    assert np.allclose(state_predictor.fields.V_star, 0.2)
    assert np.allclose(state_predictor.fields.W_star, 0.3)

def test_predict_velocity_nu_debug_format(state_predictor, capsys):
    from src.step3.predictor import predict_velocity
    predict_velocity(state_predictor)
    captured = capsys.readouterr()
    assert "Nu=1.000000e-03" in captured.out
    assert "dt=0.1" in captured.out

def test_predict_velocity_v_w_components_explicit(state_predictor):
    from src.step3.predictor import predict_velocity
    state_predictor.config.external_forces.force_vector = [0.0, 5.0, 10.0]
    predict_velocity(state_predictor)
    assert np.allclose(state_predictor.fields.V_star, 0.5)
    assert np.allclose(state_predictor.fields.W_star, 1.0)
