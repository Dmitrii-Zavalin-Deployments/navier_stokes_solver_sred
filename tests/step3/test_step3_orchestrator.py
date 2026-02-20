# tests/step3/test_orchestrate_step3.py

import pytest
import numpy as np
from src.solver_state import SolverState
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_orchestrate_step3_minimal():
    """Verify renamed orchestrator successfully updates state fields and history."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    # Provide a simple BC config to ensure orchestrator handles it
    state.config["boundary_conditions"] = [
        {"location": "x_min", "type": "no-slip"}
    ]

    result = orchestrate_step3(
        state=state,
        current_time=0.1,
        step_index=1,
    )

    assert isinstance(result, SolverState)
    assert result.fields["P"].shape == (3, 3, 3)
    
    # Check that history was appended
    assert len(result.history["times"]) == 1
    assert result.history["times"][0] == 0.1

def test_orchestrate_step3_history_persistence():
    """Ensure history builds up correctly over multiple calls."""
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)
    
    state = orchestrate_step3(state, 0.1, 1)
    state = orchestrate_step3(state, 0.2, 2)

    assert len(state.history["times"]) == 2
    assert state.history["times"] == [0.1, 0.2]

def test_orchestrate_step3_resilience():
    """Ensure orchestrator handles cases where optional state attributes are missing."""
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)
    
    # Clear optional attributes to test initialization logic
    if hasattr(state, "history"): 
        delattr(state, "history")
    state.boundary_conditions = None

    result = orchestrate_step3(state, 0.0, 0)
    
    assert "times" in result.history
    assert result.fields["U"].shape == (3, 2, 2)