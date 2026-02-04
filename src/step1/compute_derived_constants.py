# file: step1/compute_derived_constants.py
from __future__ import annotations

from typing import Dict

from .types import DerivedConstants, GridConfig


def compute_derived_constants(
    grid_config: GridConfig,
    fluid_properties: Dict[str, float],
    simulation_parameters: Dict[str, float],
) -> DerivedConstants:
    """
    Precompute physical and numerical constants for fast stencil operations.
    """
    rho = float(fluid_properties["density"])
    mu = float(fluid_properties["viscosity"])
    dt = float(simulation_parameters["time_step"])

    dx, dy, dz = grid_config.dx, grid_config.dy, grid_config.dz

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dz = 1.0 / dz

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 1.0 / (dz * dz)

    return DerivedConstants(
        rho=rho,
        mu=mu,
        dt=dt,
        dx=dx,
        dy=dy,
        dz=dz,
        inv_dx=inv_dx,
        inv_dy=inv_dy,
        inv_dz=inv_dz,
        inv_dx2=inv_dx2,
        inv_dy2=inv_dy2,
        inv_dz2=inv_dz2,
    )
