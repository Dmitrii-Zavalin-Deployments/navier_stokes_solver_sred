import pytest
from jsonschema import ValidationError

from src.step1.construct_simulation_state import construct_simulation_state
from src.step1.validate_output_schema import validate_output_schema


def test_step1_output_schema_valid(sample_json_input):
    """
    Ensure that a valid Step 1 SimulationState passes validation
    against step1_output_schema.json.
    """
    state = construct_simulation_state(sample_json_input)

    # Should NOT raise
    validate_output_schema(state)


def test_step1_output_schema_invalid_corrupted_state(sample_json_input):
    """
    Corrupt the output state and ensure validation fails.
    This confirms that step1_output_schema.json is enforced.
    """
    state = construct_simulation_state(sample_json_input)

    # Convert to dict to corrupt it
    from dataclasses import asdict, replace
    state_dict = asdict(state)

    # Remove a required field (e.g., "P")
    del state_dict["P"]

    # Re-wrap into a dummy object with missing field
    class Dummy:
        pass

    corrupted = Dummy()
    for k, v in state_dict.items():
        setattr(corrupted, k, v)

    # Expect failure
    with pytest.raises(ValidationError):
        validate_output_schema(corrupted)
