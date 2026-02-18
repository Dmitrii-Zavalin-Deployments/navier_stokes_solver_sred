# src/step1/apply_initial_conditions.py

from __future__ import annotations
from typing import Dict, Any
import numpy as np


def apply_initial_conditions(fields: Dict[str, Any], initial_conditions: Dict[str, Any]) -> None:
    """
    Apply uniform initial velocity and pressure values to the allocated fields.

    Step 1 dummy initializes fields to zeros. This function overrides them only
    if initial conditions are provided in the input. It modifies the fields dict
    in-place.

    Expected structure (from external schema):
      {
        "velocity": [u0, v0, w0],
        "pressure": p0
      }
    """

    # --- Pressure ---
    if "pressure" in initial_conditions:
        p0 = float(initial_conditions["pressure"])
        fields["P"][...] = p0

    # --- Velocity ---
    if "velocity" in initial_conditions:
        velocity = initial_conditions["velocity"]

        if not isinstance(velocity, (list, tuple)) or len(velocity) != 3:
            raise ValueError(
                f"velocity must be a 3-element list or tuple, got {velocity}"
            )

        u0, v0, w0 = [float(x) for x in velocity]

        fields["U"][...] = u0
        fields["V"][...] = v0
        fields["W"][...] = w0
