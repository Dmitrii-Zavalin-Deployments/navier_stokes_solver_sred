# file: step1/parse_config.py
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
    Extract domain, fluid, simulation, forces, BCs, and geometry into a structured Config.

    Enforces:
      • all sections are dictionaries
      • no unknown top-level keys
      • geometry_definition contains required keys
      • boundary_conditions is a list
      • external_forces values are finite numbers
    """

    # ---------------------------------------------------------
    # Validate top-level keys
    # ---------------------------------------------------------
    required_keys = {
        "domain_definition",
        "fluid_properties",
        "simulation_parameters",
        "geometry_definition",
    }

    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key: {key}")

    # ---------------------------------------------------------
    # Extract and validate dict sections
    # ---------------------------------------------------------
    domain = _ensure_dict("domain_definition", data["domain_definition"])
    fluid = _ensure_dict("fluid_properties", data["fluid_properties"])
    simulation = _ensure_dict("simulation_parameters", data["simulation_parameters"])
    geometry = _ensure_dict("geometry_definition", data["geometry_definition"])

    # ---------------------------------------------------------
    # Validate geometry_definition structure
    # ---------------------------------------------------------
    required_geom_keys = {
        "geometry_mask_flat",
        "geometry_mask_shape",
        "mask_encoding",
        "flattening_order",
    }

    for key in required_geom_keys:
        if key not in geometry:
            raise KeyError(f"geometry_definition missing required key: {key}")

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

    for k, v in forces.items():
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"external_forces[{k!r}] must be a finite number, got {v}")

    # ---------------------------------------------------------
    # Construct Config object
    # ---------------------------------------------------------
    return Config(
        domain=domain,
        fluid=fluid,
        simulation=simulation,
        forces=forces,
        boundary_conditions=bcs,
        geometry_definition=geometry,
    )
