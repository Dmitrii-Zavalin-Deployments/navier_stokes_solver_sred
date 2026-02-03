from jsonschema import ValidationError
from .validate_json_schema import validate_json_schema


def validate_input_schema(data: dict) -> None:
    """
    Public schema validation entry point used by tests.
    Delegates to validate_json_schema and raises jsonschema.ValidationError.
    """
    try:
        validate_json_schema(data)
    except ValidationError:
        raise
