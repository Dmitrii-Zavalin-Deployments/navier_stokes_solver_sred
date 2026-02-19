# src/step1/validate_physical_constraints.py
from __future__ import annotations

from typing import Any, Dict
import math
import numpy as np


def _ensure_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite number > 0, got {value}")


def _ensure_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value}")


def _ensure_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def _ensure_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def validate_physical_constraints(data: Dict[str, Any]) -> None:
    """
    Fatal physical checks before any allocation.
    Strictly validates required properties without injecting default values.
    """

    # 1. Required Sections (Will raise KeyError if missing, as expected)
    domain = data["domain"]
    fluid = data["fluid_properties"]
    init = data["initial_conditions"]
    sim = data["simulation_parameters"]
    mask = data["mask"]  # Top-level 'mask' from schema
    forces = data["external_forces"]

    # 2. Fluid properties
    _ensure_positive("density", float(fluid["density"]))
    _ensure_non_negative("viscosity", float(fluid["viscosity"]))

    # 3. Grid counts
    nx = int(domain["nx"])
    ny = int(domain["ny"])
    nz = int(domain["nz"])
    _ensure_positive_int("nx", nx)
    _ensure_positive_int("ny", ny)
    _ensure_positive_int("nz", nz)

    # 4. Domain extents (Strict access, no defaults)
    x_min, x_max = float(domain["x_min"]), float(domain["x_max"])
    y_min, y_max = float(domain["y_min"]), float(domain["y_max"])
    z_min, z_max = float(domain["z_min"]), float(domain["z_max"])

    for name, val in [
        ("x_min", x_min), ("x_max", x_max),
        ("y_min", y_min), ("y_max", y_max),
        ("z_min", z_min), ("z_max", z_max)
    ]:
        _ensure_finite(name, val)

    if x_max <= x_min: raise ValueError(f"x_max ({x_max}) must be > x_min ({x_min})")
    if y_max <= y_min: raise ValueError(f"y_max ({y_max}) must be > y_min ({y_min})")
    if z_max <= z_min: raise ValueError(f"z_max ({z_max}) must be > z_min ({z_min})")

    # 5. Mask consistency
    # Ensure mask size matches nx*ny*nz. 
    # Works for both flat lists and nested 3D lists.
    mask_size = np.array(mask).size
    if mask_size != nx * ny * nz:
        raise ValueError(
            f"Mask size {mask_size} does not match domain nx*ny*nz={nx*ny*nz}"
        )

    # 6. Initial Conditions
    vel = init["velocity"]
    if not isinstance(vel, (list, tuple)) or len(vel) != 3:
        raise ValueError("initial_conditions.velocity must be a list/tuple of length 3")
    
    for i, v in enumerate(vel):
        _ensure_finite(f"initial_conditions.velocity[{i}]", float(v))
    
    _ensure_finite("initial_conditions.pressure", float(init["pressure"]))

    # 7. Time step
    _ensure_positive("time_step", float(sim["time_step"]))

    # 8. External forces
    fv = forces["force_vector"]
    if not isinstance(fv, (list, tuple)) or len(fv) != 3:
        raise ValueError("external_forces.force_vector must be a list/tuple of length 3")
    
    for i, comp in enumerate(fv):
        _ensure_finite(f"external_forces.force_vector[{i}]", float(comp))