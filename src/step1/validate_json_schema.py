import json
from pathlib import Path
from jsonschema import validate, ValidationError


# Resolve path to schema/input_schema.json
SCHEMA_PATH = (
    Path(__file__)
    .resolve()
    .parents[2]  # go from src/step1/ → src/ → project root
    / "schema"
    / "input_schema.json"
)

# Load schema once at import time
with open(SCHEMA_PATH, "r") as f:
    INPUT_SCHEMA = json.load(f)


def validate_json_schema(data: dict) -> None:
    """
    Validate raw Step 1 input JSON against schema/input_schema.json.
    Raises jsonschema.ValidationError on failure.
    """
    try:
        validate(instance=data, schema=INPUT_SCHEMA)
    except ValidationError:
        # Re-raise exactly as tests expect
        raise
