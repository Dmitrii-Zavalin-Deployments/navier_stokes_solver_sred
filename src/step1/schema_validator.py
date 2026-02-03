import json
import os
from jsonschema import validate, Draft7Validator, ValidationError


# Path to the schema file inside the repository
SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "schema",
    "input_schema.json"
)


def load_schema():
    """Load the JSON schema from the schema/ directory."""
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    with open(SCHEMA_PATH, "r") as f:
        return json.load(f)


def validate_input_schema(data):
    """
    Validate the input JSON against the schema.
    Raises jsonschema.ValidationError on failure.
    """
    schema = load_schema()

    # Create a validator instance (Draft-07)
    validator = Draft7Validator(schema)

    # Collect all validation errors (if any)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)

    if errors:
        # Raise the first error for simplicity
        raise ValidationError(errors[0].message)

    # If no errors, validation succeeded
    return True
