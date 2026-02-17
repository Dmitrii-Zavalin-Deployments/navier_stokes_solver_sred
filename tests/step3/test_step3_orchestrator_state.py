# tests/step3/test_step3_orchestrator_state.py

from src.solver_state import SolverState
from src.step3.orchestrate_step3 import orchestrate_step3_state
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_orchestrate_step3_state_minimal():
    """
    Minimal structural test for the Step‑3 orchestrator.
    Ensures that Step‑3 updates fields, health, and history on a valid Step‑2 state.
    """
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)

    result = orchestrate_step3_state(
        state=state,
        current_time=0.0,
        step_index=0,
    )

    assert isinstance(result, SolverState)
    assert "P" in result.fields
    assert result.fields["P"].shape == (2, 2, 2)

    # Step‑3 must populate health and history
    assert isinstance(result.health, dict)
    assert isinstance(result.history, dict)

    assert "times" in result.history
    assert "divergence_norms" in result.history
    assert "max_velocity_history" in result.history


def test_step3_optional_fields():
    """
    Step‑3 must tolerate optional or empty fields in Step‑2 state.
    """
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)

    # Remove optional fields to ensure Step‑3 handles them gracefully
    state.history = {}
    state.health = {}
    state.boundary_conditions = None

    result = orchestrate_step3_state(
        state=state,
        current_time=0.0,
        step_index=0,
    )

    assert isinstance(result, SolverState)
    assert "P" in result.fields
