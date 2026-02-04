# file: step1/validate_json_schema.py
from __future__ import annotations

from typing import Any, Dict

import jsonschema


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Ensure all required keys exist and have correct types and structures.
    Structural-only: no physics, no semantics.
    """
    jsonschema.validate(instance=data, schema=schema)
