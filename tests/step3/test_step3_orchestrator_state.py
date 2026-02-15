# tests/step3/test_step3_orchestrator_state.py

from src.solver_state import SolverState
from src.step3.orchestrate_step3 import orchestrate_step3_state


def test_orchestrate_step3_state_minimal():
    """
    Minimal structural test for the new state-based Step 3 orchestrator.
    Ensures that Step 3 updates fields, health, and history on SolverState.
    """

    # Minimal dummy Step-2 output (replace with your helper if available)
    dummy = {
        "config": {
            "domain": {"nx": 2, "ny": 2, "nz": 2},
            "dt": 0.1,
        },
        "grid": {
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
        },
        "fields": {
            "P": None,  # Step 3 will compute pressure
            "U": None,
            "V": None,
            "W": None,
        },
        "mask": None,
        "constants": {"rho": 1.0},
        "boundary_conditions": {},
        "health": {},
        "ppe_structure": {},
        "operators": {},
    }

    # Build SolverState from dummy Step-2 output
    state = SolverState()
    state.config = dummy["config"]
    state.grid = dummy["grid"]
    state.fields = dummy["fields"]
    state.mask = dummy["mask"]
    state.constants = dummy["constants"]
    state.boundary_conditions = dummy["boundary_conditions"]
    state.health = dummy["health"]
    state.ppe = dummy["ppe_structure"]
    state.operators = dummy["operators"]

    # Run Step 3 (state-based)
    state = orchestrate_step3_state(
        state,
        current_time=0.0,
        step_index=0,
    )

    # ---------------------------------------------------------
    # Assertions: Step 3 must update fields, health, history
    # ---------------------------------------------------------
    assert state.fields is not None
    assert "P" in state.fields  # Pressure must be computed
    assert state.health is not None
    assert isinstance(state.history, dict)

    # Optional: check some expected history keys
    assert "times" in state.history
    assert "divergence_norms" in state.history
    assert "max_velocity_history" in state.history
