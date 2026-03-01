# src/step1/apply_initial_conditions.py

from __future__ import annotations
from src.solver_input import InitialConditionsInput
from typing import Dict, Any
import numpy as np

def apply_initial_conditions(fields: Dict[str, np.ndarray], initial_conditions: InitialConditionsInput) -> None:
    """
    Primes the allocated fields with uniform values from the configuration.

    Constitutional Role: Field Primer.
    Requirement: Ensures 'Loud' values from JSON are successfully ingested.
    
    Args:
        fields: Dictionary of NumPy arrays (P, U, V, W).
        initial_conditions: Config fragment containing 'velocity' and 'pressure'.
    """

    # --- Pressure Priming ---
    if "pressure" in initial_conditions:
        try:
            p0 = float(initial_conditions.pressure)
            fields["P"].fill(p0)
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"Invalid pressure initial condition: {e}")

    # --- Velocity Priming ---
    if "velocity" in initial_conditions:
        velocity = initial_conditions.velocity

        if not isinstance(velocity, (list, tuple)) or len(velocity) != 3:
            raise ValueError(
                f"Initial velocity must be a 3-element list [u, v, w], got {velocity}"
            )

        try:
            # Map components and apply to staggered fields
            u0, v0, w0 = [float(x) for x in velocity]
            
            fields["U"].fill(u0)
            fields["V"].fill(v0)
            fields["W"].fill(w0)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not cast initial velocity components to float: {e}")