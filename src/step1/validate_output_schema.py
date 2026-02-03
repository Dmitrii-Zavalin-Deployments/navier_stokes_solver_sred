import json
from pathlib import Path
from dataclasses import asdict
from jsonschema import validate, ValidationError


# Path to schema/step1_output_schema.json
SCHEMA_PATH = (
    Path(__file__)
    .resolve()
    .parents[2]  # step1/ → src/ → project root
    / "schema"
    / "step1_output_schema.json"
)

# Load schema once at import time
with open(SCHEMA_PATH, "r") as f:
    STEP1_OUTPUT_SCHEMA = json.load(f)


def validate_output_schema(state_obj) -> None:
    """
    Validate the final SimulationState produced by Step 1
    against schema/step1_output_schema.json.

    Raises jsonschema.ValidationError on failure.
    """
    # Convert dataclass → dict for JSON Schema validation
    data = asdict(state_obj)

    try:
        validate(instance=data, schema=STEP1_OUTPUT_SCHEMA)
    except ValidationError:
        # Re-raise exactly as tests expect
        raise
