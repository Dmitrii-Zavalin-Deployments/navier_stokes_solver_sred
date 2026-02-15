# tests/contract/test_run_solver_minimal.py

from src.main_solver import run_solver
from src.solver_state import SolverState


def test_run_solver_minimal():
    config = {
        "domain": {"nx": 2, "ny": 2, "nz": 2},
        "initial_conditions": {
            "initial_pressure": 0.0,
            "initial_velocity": [0.0, 0.0, 0.0]
        },
        "boundary_conditions": []
    }

    state = run_solver(config)

    # Basic structural checks
    assert isinstance(state, SolverState)
    assert state.config == config

    assert state.grid
    assert state.fields
    assert state.mask is not None

    # Extended fields from Step 4
    assert state.P_ext is not None
    assert state.U_ext is not None
    assert state.V_ext is not None
    assert state.W_ext is not None

    assert state.ready_for_time_loop is True
