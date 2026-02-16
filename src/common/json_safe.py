# src/common/json_safe.py
"""
Shared JSON‑safe conversion utility.

This module provides a single canonical function `to_json_safe`
that recursively converts:
    • NumPy arrays → Python lists
    • dicts → dicts with JSON‑safe values
    • lists/tuples → lists with JSON‑safe values

It is used by orchestrators (Steps 1–4) and by schema validation.
"""

from __future__ import annotations
from typing import Any
import numpy as np


def to_json_safe(obj: Any) -> Any:
    """
    Recursively convert solver state objects into JSON‑safe structures.

    Rules:
      • NumPy arrays → .tolist()
      • dict → recursively convert values
      • list/tuple → recursively convert elements
      • everything else → returned unchanged

    This ensures schema validation and serialization never choke
    on NumPy arrays or solver‑internal objects.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]

    return obj
