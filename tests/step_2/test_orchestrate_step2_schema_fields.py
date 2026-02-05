# tests/step_2/test_orchestrate_step2_schema_fields.py

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.dummy_state_step2 import DummyState


def test_orchestrate_step2_schema_fields():
    state = DummyState(4, 4, 4)
    result = orchestrate_step2(state)

    required = [
        "Constants",
        "Mask",
        "is_fluid",
        "is_boundary_cell",
        "Operators",
        "PPE",
        "Health",
        "AdvectionMeta",
    ]

    for key in required:
        assert key in result, f"Missing required field: {key}"
