# tests/step3/test_step3_integration.py

import numpy as np
from src.step3.orchestrate_step3 import orchestrate_step3_state
from src.solver_state import SolverState


def _make_minimal_state(nx: int = 3, ny: int = 3, nz: int = 3) -> SolverState:
    state = SolverState()
    state.config = {}
    state.grid = {"nx": nx, "ny": ny, "nz": nz}
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }
    state.mask = np.ones((nx, ny, nz), dtype=int)
    state.constants = {"rho": 1.0}
    state.boundary_conditions = {}
    state.health = {}
    state.ppe = {}
    state.operators = {}
    return state


def test_step3_integration_minimal():
    state = _make_minimal_state()

    result = orchestrate_step3_state(state, current_time=0.0, step_index=0)

    assert isinstance(result, SolverState)
    assert "P" in result.fields
    assert result.fields["P"].shape == (3, 3, 3)


def test_step3_optional_fields():
    state = _make_minimal_state()

    result = orchestrate_step3_state(state, current_time=0.0, step_index=0)

    assert isinstance(result, SolverState)
    assert "P" in result.fields
