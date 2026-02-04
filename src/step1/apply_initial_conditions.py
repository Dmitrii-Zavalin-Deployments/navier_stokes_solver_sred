# file: step1/apply_initial_conditions.py
from __future__ import annotations

from typing import Dict

import numpy as np

from .types import Fields


def apply_initial_conditions(fields: Fields, initial_conditions: Dict[str, object]) -> None:
    """
    Populate P, U, V, W with uniform initial values.
    In-place operation.
    """
    p0 = float(initial_conditions["initial_pressure"])
    u0, v0, w0 = [float(x) for x in initial_conditions["initial_velocity"]]

    fields.P[...] = p0
    fields.U[...] = u0
    fields.V[...] = v0
    fields.W[...] = w0

    # Mask is left as allocated (structural only in Step 1).
