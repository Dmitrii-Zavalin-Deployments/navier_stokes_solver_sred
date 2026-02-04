# file: step1/parse_config.py
from __future__ import annotations

from typing import Any, Dict

from .types import Config


def parse_config(data: Dict[str, Any]) -> Config:
    """
    Extract domain, fluid, simulation, forces, BCs, geometry into a structured Config.
    """
    domain = dict(data["domain_definition"])
    fluid = dict(data["fluid_properties"])
    simulation = dict(data["simulation_parameters"])
    forces = dict(data.get("external_forces", {}))
    bcs = list(data.get("boundary_conditions", []))
    geometry = dict(data["geometry_definition"])

    return Config(
        domain=domain,
        fluid=fluid,
        simulation=simulation,
        forces=forces,
        boundary_conditions=bcs,
        geometry_definition=geometry,
    )
