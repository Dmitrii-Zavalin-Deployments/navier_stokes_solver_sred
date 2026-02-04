# file: step1/apply_initial_conditions.py
from __future__ import annotations

from typing import Dict
import math

from .types import Fields


def apply_initial_conditions(fields: Fields, initial_conditions: Dict[str, object]) -> None:
    """
    Populate P, U, V, W with uniform initial values.
    In-place operation.
    """
    p0 = float(initial_conditions["initial_pressure"])
    u0, v0, w0 = [float(x) for x in initial_conditions["initial_velocity"]]

    # NEW: enforce finite velocity components
    for name, val in zip(["u", "v", "w"], [u0, v0, w0]):
        if not math.isfinite(val):
            raise ValueError(f"Initial velocity must contain finite numbers, got {name}={val}")

    # NEW: enforce finite pressure
    if not math.isfinite(p0):
        raise ValueError(f"Initial pressure must be finite, got {p0}")

    fields.P[...] = p0
    fields.U[...] = u0
    fields.V[...] = v0
    fields.W[...] = w0
