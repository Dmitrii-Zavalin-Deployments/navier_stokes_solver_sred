# file: step1/apply_initial_conditions.py
from __future__ import annotations

from typing import Dict
import math

from .types import Fields


def apply_initial_conditions(fields: Fields, initial_conditions: Dict[str, object]) -> None:
    """
    Populate P, U, V, W with uniform initial values (cell-centered).
    In-place operation.

    Step 1 has already validated the input schema, so here we enforce
    only numerical safety (finite values, correct vector length).
    """

    # --- Validate presence of required keys ---
    if "initial_pressure" not in initial_conditions:
        raise KeyError("Missing required key: initial_pressure")

    if "initial_velocity" not in initial_conditions:
        raise KeyError("Missing required key: initial_velocity")

    velocity = initial_conditions["initial_velocity"]

    # --- Validate velocity vector shape ---
    if not isinstance(velocity, (list, tuple)) or len(velocity) != 3:
        raise ValueError(
            f"initial_velocity must be a 3-element list or tuple, got {velocity}"
        )

    # --- Extract and validate values ---
    p0 = float(initial_conditions["initial_pressure"])
    u0, v0, w0 = [float(x) for x in velocity]

    # Finite checks
    if not math.isfinite(p0):
        raise ValueError(f"Initial pressure must be finite, got {p0}")

    for name, val in zip(["u", "v", "w"], [u0, v0, w0]):
        if not math.isfinite(val):
            raise ValueError(f"Initial velocity must contain finite numbers, got {name}={val}")

    # --- Apply initial conditions (cell-centered arrays) ---
    fields.P[...] = p0
    fields.U[...] = u0
    fields.V[...] = v0
    fields.W[...] = w0
