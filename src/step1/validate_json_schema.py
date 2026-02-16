# file: src/step1/validate_json_schema.py
from __future__ import annotations

from typing import Any, Dict

from .schema_utils import validate_with_schema


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Thin wrapper around the strict Draft 2020‑12 JSON Schema validator.

    Step 1 responsibilities:
      • structural validation only
      • no interpretation of geometry, BCs, or solver semantics
      • no MAC‑grid or cell‑centered logic here

    All validation is delegated to schema_utils.validate_with_schema.
    This wrapper exists for backward compatibility and clarity.
    """
    validate_with_schema(data, schema)
