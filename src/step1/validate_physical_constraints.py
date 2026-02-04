# file: step1/validate_physical_constraints.py
from __future__ import annotations

from typing import Any, Dict

import math


def _ensure_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _ensure_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _ensure_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def _ensure_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def validate_physical_constraints(data: Dict[str, Any]) -> None:
    """
    Fatal physical checks before any allocation.
    Structural + basic physical only.
    """
    domain = data["domain_definition"]
    fluid = data["fluid_properties"]
    init = data["initial_conditions"]
    geom = data["geometry_definition"]

    # Density, viscosity
    _ensure_positive("density", float(fluid["density"]))
    _ensure_non_negative("viscosity", float(fluid["viscosity"]))

    # Grid counts
    nx = int(domain["nx"])
    ny = int(domain["ny"])
    nz = int(domain["nz"])
    _ensure_positive_int("nx", nx)
    _ensure_positive_int("ny", ny)
    _ensure_positive_int("nz", nz)

    # Domain extents
    x_min = float(domain["x_min"])
    x_max = float(domain["x_max"])
    y_min = float(domain["y_min"])
    y_max = float(domain["y_max"])
    z_min = float(domain["z_min"])
    z_max = float(domain["z_max"])

    if not x_max > x_min:
        raise ValueError("x_max must be > x_min")
    if not y_max > y_min:
        raise ValueError("y_max must be > y_min")
    if not z_max > z_min:
        raise ValueError("z_max must be > z_min")

    # Mask length
    mask_flat = geom["geometry_mask_flat"]
    if len(mask_flat) != nx * ny * nz:
        raise ValueError(
            f"geometry_mask_flat length {len(mask_flat)} "
            f"does not match nx*ny*nz={nx*ny*nz}"
        )

    # Initial velocity: length and finiteness
    vel = init["initial_velocity"]
    if len(vel) != 3:
        raise ValueError("initial_velocity must have exactly 3 components")
    for i, v in enumerate(vel):
        _ensure_finite(f"initial_velocity[{i}]", float(v))

    # Initial pressure
    _ensure_finite("initial_pressure", float(init["initial_pressure"]))

    # Optional CFL pre-check (very light)
    sim = data["simulation_parameters"]
    dt = float(sim["time_step"])
    _ensure_positive("time_step", dt)
    dx = (x_max - x_min) / nx
    max_u = max(abs(float(v)) for v in vel)
    if max_u > 0.0 and dt * max_u > dx:
        raise ValueError("CFL pre-check failed: dt * |u| > dx")
