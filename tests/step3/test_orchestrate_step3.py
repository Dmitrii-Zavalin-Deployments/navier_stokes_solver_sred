# tests/step3/test_orchestrate_step3.py

import pytest
import numpy as np
from src.solver_state import SolverState
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def _prepare_mock_state(nx):
    state = make_step2_output_dummy(nx=nx, ny=nx, nz=nx)
    # Inject gradients to avoid KeyError in correction step
    state.operators["grad_x"] = lambda P: np.zeros_like(state.fields["U"])
    state.operators["grad_y"] = lambda P: np.zeros_like(state.fields["V"])
    state.operators["grad_z"] = lambda P: np.zeros_like(state.fields["W"])
    return state

def test_orchestrate_step3_minimal():
    state = _prepare_mock_state(3)
    state.config["boundary_conditions"] = [{"location": "x_min", "type": "no-slip"}]

    result = orchestrate_step3(state=state, current_time=0.1, step_index=1)

    assert isinstance(result, SolverState)
    assert result.fields["P"].shape == (3, 3, 3)
    assert len(result.history["times"]) == 1

def test_orchestrate_step3_history_persistence():
    state = _prepare_mock_state(2)
    state = orchestrate_step3(state, 0.1, 1)
    state = orchestrate_step3(state, 0.2, 2)

    assert state.history["times"] == [0.1, 0.2]

def test_orchestrate_step3_resilience():
    state = _prepare_mock_state(2)
    if hasattr(state, "history"): 
        delattr(state, "history")
    state.boundary_conditions = None

    result = orchestrate_step3(state, 0.0, 0)
    
    assert hasattr(result, "history")
    assert "times" in result.history
    assert result.fields["U"].shape == (3, 2, 2)