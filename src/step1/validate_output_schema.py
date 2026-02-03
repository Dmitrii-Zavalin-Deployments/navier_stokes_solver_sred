import json
from pathlib import Path
from dataclasses import asdict, is_dataclass
from jsonschema import validate, ValidationError
import numpy as np


# Path to schema/step1_output_schema.json
SCHEMA_PATH = (
    Path(__file__)
    .resolve()
    .parents[2]
    / "schema"
    / "step1_output_schema.json"
)

with open(SCHEMA_PATH, "r") as f:
    STEP1_OUTPUT_SCHEMA = json.load(f)


def _convert_numpy(obj):
    """
    Recursively convert numpy arrays to Python lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(x) for x in obj]
    return obj


def validate_output_schema(state_obj) -> None:
    """
    Validate the final SimulationState produced by Step 1
    against schema/step1_output_schema.json.
    Converts numpy arrays to lists before validation.
    Supports both dataclass instances and dict-like objects.
    """
    # Convert dataclass → dict, or fallback to __dict__
    if is_dataclass(state_obj):
        data = asdict(state_obj)
    else:
        # Used in corrupted-state tests
        data = dict(state_obj.__dict__)

    # Convert numpy arrays → lists
    data = _convert_numpy(data)

    try:
        validate(instance=data, schema=STEP1_OUTPUT_SCHEMA)
    except ValidationError:
        raise
