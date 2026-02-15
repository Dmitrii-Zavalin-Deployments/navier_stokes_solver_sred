# tests/step4/test_step4_orchestrator_state.py

from src.solver_state import SolverState
from src.step4.orchestrate_step4 import orchestrate_step4_state
from tests.helpers.step3_schema_dummy_state import STEP3_MINIMAL_STATE


def test_orchestrate_step4_state_minimal():
    dummy = STEP3_MINIMAL_STATE  # dict with config, fields, mask, health

    state = SolverState()
    state.config = dummy["config"]
    state.fields = dummy["fields"]
    state.mask = dummy["mask"]
    state.health = dummy["health"]

    state = orchestrate_step4_state(state)

    assert state.P_ext is not None
    assert state.U_ext is not None
    assert state.V_ext is not None
    assert state.W_ext is not None

    assert state.ready_for_time_loop is True
    assert isinstance(state.step4_diagnostics, dict)
