import json
from pathlib import Path
from dataclasses import asdict, is_dataclass
from jsonschema import validate
import numpy as np

SCHEMA_PATH = (
    Path(__file__)
    .resolve()
    .parents[1]
    / ".."
    / "schema"
    / "step1_output_schema.json"
).resolve()

with open(SCHEMA_PATH, "r") as f:
    STEP1_OUTPUT_SCHEMA = json.load(f)

def _convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(x) for x in obj]
    return obj

def validate_output_schema(state_obj) -> None:
    # Convert object → dict
    if is_dataclass(state_obj):
        data = asdict(state_obj)
    elif hasattr(state_obj, "__dict__"):
        data = dict(state_obj.__dict__)
    elif isinstance(state_obj, dict):
        data = state_obj
    else:
        try:
            data = dict(state_obj)
        except Exception:
            raise TypeError("Unsupported state object type")

    # Convert numpy arrays → lists
    data = _convert_numpy(data)

    # Validate against JSON Schema
    validate(instance=data, schema=STEP1_OUTPUT_SCHEMA)
