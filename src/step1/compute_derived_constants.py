# file: step1/compute_derived_constants.py

from __future__ import annotations

from typing import Dict
import math

from .types import DerivedConstants, GridConfig


def compute_derived_constants(
    grid_config: GridConfig,
    fluid_properties: Dict[str, float],
    simulation_parameters: Dict[str, float],
) -> DerivedConstants:
    """
    Compute physical and numerical constants for Step 1.

    Step 1 responsibilities:
      - validate positivity / finiteness of physical parameters
      - compute dx, dy, dz and their inverses
      - compute dt and its inverses
      - produce a DerivedConstants object used by all later steps

    This function is fully aligned with the cell-centered solver architecture.
    No MAC-grid or staggered assumptions appear here.
    """

    # -----------------------------
    # Validate required keys
    # -----------------------------
    for key in ["density", "viscosity"]:
        if key not in fluid_properties:
            raise KeyError(f"Missing required fluid property: {key}")

    if "time_step" not in simulation_parameters:
        raise KeyError("Missing required simulation parameter: time_step")

    # -----------------------------
    # Extract and validate values
    # -----------------------------
    rho = float(fluid_properties["density"])
    mu = float(fluid_properties["viscosity"])
    dt = float(simulation_parameters["time_step"])

    if not (math.isfinite(rho) and rho > 0):
        raise ValueError(f"Density must be a finite positive number, got {rho}")

    if not (math.isfinite(mu) and mu >= 0):
        raise ValueError(f"Viscosity must be a finite non-negative number, got {mu}")

    if not (math.isfinite(dt) and dt > 0):
        raise ValueError(f"Time step must be a finite positive number, got {dt}")

    dx, dy, dz = grid_config.dx, grid_config.dy, grid_config.dz

    for name, val in [("dx", dx), ("dy", dy), ("dz", dz)]:
        if not (math.isfinite(val) and val > 0):
            raise ValueError(f"{name} must be a finite positive number, got {val}")

    # -----------------------------
    # Compute derived constants
    # -----------------------------
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
