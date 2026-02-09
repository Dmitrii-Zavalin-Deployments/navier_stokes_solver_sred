# file: src/step1/schema_utils.py
import json
from pathlib import Path
from functools import lru_cache
from jsonschema import Draft202012Validator, ValidationError
import numpy as np


# ---------------------------------------------------------
# Helper: make instances JSON/schema‑safe
# ---------------------------------------------------------
def _to_json_safe(obj):
    """Recursively convert NumPy arrays to Python lists for JSON/schema compatibility."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    return obj


# ---------------------------------------------------------
# Load and cache schemas
# ---------------------------------------------------------
@lru_cache(maxsize=None)
def load_schema(path: str) -> dict:
    schema_path = Path(path)
    with open(schema_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------
# Validate instance against schema with strict Draft 2020-12
# ---------------------------------------------------------
def validate_with_schema(instance: dict, schema: dict) -> None:
    """
    Validate a JSON-like instance against a Draft 2020-12 schema.

    Provides:
      • strict validation
      • clear error messages
      • full path to failing field
    """

    if not isinstance(instance, dict):
        raise TypeError(
            f"Schema validation requires a dict instance, got {type(instance).__name__}"
        )

    # Make a JSON‑safe copy (handles mask_3d, fields, etc.)
    instance_safe = _to_json_safe(instance)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance_safe), key=lambda e: e.path)

    if errors:
        err = errors[0]  # show the first, most relevant error

        path = " → ".join(str(p) for p in err.path) or "<root>"
        expected = err.schema.get("type", "unknown")
        message = (
            f"Schema validation error at: {path}\n"
            f"Message: {err.message}\n"
            f"Expected type: {expected}\n"
            f"Validator: {err.validator}\n"
            f"Validator value: {err.validator_value}\n"
        )

        raise ValidationError(message)
