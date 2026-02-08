# file: step1/validate_json_schema.py
from __future__ import annotations

from typing import Any, Dict

from .schema_utils import validate_with_schema


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Thin wrapper around the strict Draft 2020-12 validator.

    This function exists for backward compatibility only.
    All validation is delegated to schema_utils.validate_with_schema.
    """
    validate_with_schema(data, schema)
