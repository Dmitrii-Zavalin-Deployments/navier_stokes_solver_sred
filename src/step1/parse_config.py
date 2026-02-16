# src/step1/parse_config.py
from __future__ import annotations

from typing import Any, Dict
import math

from .types import Config


def _ensure_dict(name: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary, got {type(value).__name__}")
    return dict(value)


def parse_config(data: Dict[str, Any]) -> Config:
    """
    Extract domain, fluid_properties, simulation_parameters,
    external_forces, boundary_conditions, and geometry into a structured Config.

    Enforces:
      • all sections are dictionaries
      • no unknown top-level keys
      • geometry contains required keys
      • boundary_conditions is a list
      • external_forces.force_vector is a finite 3‑vector
    """

    # ---------------------------------------------------------
    # Validate top-level keys (new schema)
    # ---------------------------------------------------------
    required_keys = {
        "domain",
        "fluid_properties",
        "simulation_parameters",
        "geometry",
    }

    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key: {key}")

    # ---------------------------------------------------------
    # Extract and validate dict sections
    # ---------------------------------------------------------
    domain = _ensure_dict("domain", data["domain"])
    fluid = _ensure_dict("fluid_properties", data["fluid_properties"])
    simulation = _ensure_dict("simulation_parameters", data["simulation_parameters"])
    geometry = _ensure_dict("geometry", data["geometry"])

    # ---------------------------------------------------------
    # Validate geometry structure (new schema)
    # ---------------------------------------------------------
    required_geom_keys = {
        "mask_flat",
        "mask_shape",
        "flattening_order",
    }

    for key in required_geom_keys:
        if key not in geometry:
            raise KeyError(f"geometry missing required key: {key}")

    # ---------------------------------------------------------
    # Validate boundary_conditions
    # ---------------------------------------------------------
    bcs_raw = data.get("boundary_conditions", [])
    if not isinstance(bcs_raw, list):
        raise TypeError("boundary_conditions must be a list")
    bcs = list(bcs_raw)

    # ---------------------------------------------------------
    # Validate external_forces
    # ---------------------------------------------------------
    forces_raw = data.get("external_forces", {})
    forces = _ensure_dict("external_forces", forces_raw)

    fv = forces.get("force_vector")

    if fv is None:
        raise KeyError("external_forces must contain 'force_vector'")

    # Validate force vector
    if not isinstance(fv, (list, tuple)) or len(fv) != 3:
        raise ValueError(
            f"external_forces['force_vector'] must be a length‑3 vector, got {fv}"
        )

    for x in fv:
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError(
                f"external_forces['force_vector'] entries must be finite numbers, got {fv}"
            )

    # Normalize force_vector
    forces_out = dict(forces)
    forces_out["force_vector"] = list(fv)

    # ---------------------------------------------------------
    # Construct Config object (new schema fields)
    # ---------------------------------------------------------
    return Config(
        domain=domain,
        fluid_properties=fluid,
        simulation_parameters=simulation,
        external_forces=forces_out,
        boundary_conditions=bcs,
        geometry=geometry,
    )
