# src/step1/parse_config.py

from __future__ import annotations
from typing import Any, Dict
import math


def _ensure_dict(name: str, value: Any) -> Dict[str, Any]:
    """
    Ensure a value is a dictionary. Used for validating sections of the input.
    """
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary, got {type(value).__name__}")
    return dict(value)


def parse_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the solver input dictionary into the Step 1 config structure.

    Step 1 config contains only:
      • dt
      • external_forces

    This matches the frozen Step 1 schema and Step 1 dummy.
    """

    # ---------------------------------------------------------
    # Extract simulation parameters
    # ---------------------------------------------------------
    sim = _ensure_dict("simulation_parameters", data["simulation_parameters"])
    dt = sim["time_step"]

    # ---------------------------------------------------------
    # Extract external forces
    # ---------------------------------------------------------
    forces_raw = data.get("external_forces", {})
    forces = _ensure_dict("external_forces", forces_raw)

    fv = forces.get("force_vector")
    if fv is None:
        raise KeyError("external_forces must contain 'force_vector'")

    if not isinstance(fv, (list, tuple)) or len(fv) != 3:
        raise ValueError("external_forces['force_vector'] must be a length‑3 vector")

    for x in fv:
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError("force_vector entries must be finite numbers")

    # Normalize force_vector to list
    forces_out = dict(forces)
    forces_out["force_vector"] = list(fv)

    # ---------------------------------------------------------
    # Step 1 config output (dict, not Config object)
    # ---------------------------------------------------------
    return {
        "dt": dt,
        "external_forces": forces_out,
    }
