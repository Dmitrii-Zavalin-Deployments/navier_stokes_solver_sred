# tests/step3/test_orchestrate_step3.py

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def _prepare_mock_state(nx):
    """Setup a state with all operators required for Step 3 Orchestration."""
    state = make_step2_output_dummy(nx=nx, ny=nx, nz=nx)
    
    # 1. Mock Gradients
    state.operators["grad_x"] = lambda P: np.zeros_like(state.fields["U"])
    state.operators["grad_y"] = lambda P: np.zeros_like(state.fields["V"])
    state.operators["grad_z"] = lambda P: np.zeros_like(state.fields["W"])
    
    # 2. Setup Sparse Divergence Operator
    size_p = nx**3
    total_vel = state.fields["U"].size + state.fields["V"].size + state.fields["W"].size
    state.operators["divergence"] = csr_matrix((size_p, total_vel))
    
    # 3. Setup PPE Solver Matrix A
    state.ppe["A"] = eye(size_p, format="csr")
    state.ppe["ppe_is_singular"] = False
    
    # 4. Mock Config Settings
    state.config["solver_settings"] = {
        "ppe_tolerance": 1e-6,
        "ppe_atol": 1e-12,
        "ppe_max_iter": 100
    }
    
    return state

def test_orchestrate_step3_minimal():
    """Verify renamed orchestrator successfully updates state fields and history."""
    state = _prepare_mock_state(3)
    state.config["boundary_conditions"] = [{"location": "x_min", "type": "no-slip"}]

    result = orchestrate_step3(state=state, current_time=0.1, step_index=1)

    assert isinstance(result.fields["P"], np.ndarray)
    assert len(result.history["times"]) == 1
    assert result.history["times"][0] == 0.1
    # Check that new PPE metadata is being tracked
    assert "ppe_status_history" in result.history
    assert result.history["ppe_status_history"][0] == "Success"

def test_orchestrate_step3_resilience():
    """Ensure orchestrator handles cases where history attribute is missing."""
    state = _prepare_mock_state(2)
    if hasattr(state, "history"): 
        delattr(state, "history")
    
    state.boundary_conditions = None

    result = orchestrate_step3(state, 0.5, 5)
    
    assert hasattr(result, "history")
    assert result.history["times"] == [0.5]
    assert "ppe_atol_history" in result.history