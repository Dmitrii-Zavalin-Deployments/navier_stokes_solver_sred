# tests/final_validation/test_validate_final_state_minimal.py

from src.main_solver import run_solver
from src.common.final_schema_utils import validate_final_state


def test_validate_final_state_minimal():
    config = {
        "domain": {"nx": 2, "ny": 2, "nz": 2},
        "initial_conditions": {
            "initial_pressure": 0.0,
            "initial_velocity": [0.0, 0.0, 0.0]
        },
        "boundary_conditions": []
    }

    state = run_solver(config)

    # Should not raise
    validate_final_state(state)

    # Quick structural sanity
    payload = state.to_json_safe()
    assert "fields" in payload
    assert "P_ext" in payload
    assert "ready_for_time_loop" in payload
    assert payload["ready_for_time_loop"] is True
