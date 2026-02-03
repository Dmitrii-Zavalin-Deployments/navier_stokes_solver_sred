from .validate_json_schema import validate_json_schema


def validate_input_schema(data: dict) -> None:
    """
    Public schema validation entry point used by tests.
    Delegates directly to the JSON Schema validator.
    Raises jsonschema.ValidationError on failure.
    """
    validate_json_schema(data)
