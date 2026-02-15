# tests/contract/test_run_solver_final_schema.py

from src.main_solver import run_solver
from src.common.final_schema_utils import validate_final_state
from src.solver_state import SolverState


def test_run_solver_final_schema():
    config = {
        "domain": {"nx": 2, "ny": 2, "nz": 2},
        "initial_conditions": {
            "initial_pressure": 0.0,
            "initial_velocity": [0.0, 0.0, 0.0]
        },
        "boundary_conditions": []
    }

    state = run_solver(config)

    # Type and structural checks
    assert isinstance(state, SolverState)
    assert state.ready_for_time_loop is True

    # Should not raise
    validate_final_state(state)

    # Quick payload sanity
    payload = state.to_json_safe()
    assert "fields" in payload
    assert "P_ext" in payload
    assert "U_ext" in payload
    assert "ready_for_time_loop" in payload
