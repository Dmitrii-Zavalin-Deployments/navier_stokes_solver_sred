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
    if not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def _ensure_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def validate_physical_constraints(data: Dict[str, Any]) -> None:
    """
    Fatal physical checks for the Step 1 state.
    Matches the state structure: grid, constants, fields, and mask.
    """

    # 1. Access sections as they appear in the DEBUG State Summary
    # These will raise KeyError if the orchestrator fails to provide them.
    grid = data["grid"]
    constants = data["constants"]
    fields = data["fields"]
    mask = data["mask"]

    # 2. Fluid properties & Time step (extracted from constants dictionary)
    _ensure_positive("density (rho)", float(constants["rho"]))
    _ensure_non_negative("viscosity (mu)", float(constants["mu"]))
    _ensure_positive("time_step (dt)", float(constants["dt"]))

    # 3. Grid counts & Spacing
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    _ensure_positive_int("nx", nx)
    _ensure_positive_int("ny", ny)
    _ensure_positive_int("nz", nz)
    
    _ensure_positive("dx", float(grid["dx"]))
    _ensure_positive("dy", float(grid["dy"]))
    _ensure_positive("dz", float(grid["dz"]))

    # 4. Domain extents validation
    # Using .get() because processed grid state usually prioritizes dx/dy/dz
    x_min, x_max = grid.get("x_min"), grid.get("x_max")
    y_min, y_max = grid.get("y_min"), grid.get("y_max")
    z_min, z_max = grid.get("z_min"), grid.get("z_max")

    # Validate X-axis
    if x_min is not None and x_max is not None:
        _ensure_finite("x_min", x_min)
        _ensure_finite("x_max", x_max)
        if x_max <= x_min:
            raise ValueError(f"x_max ({x_max}) must be > x_min ({x_min})")

    # Validate Y-axis
    if y_min is not None and y_max is not None:
        _ensure_finite("y_min", y_min)
        _ensure_finite("y_max", y_max)
        if y_max <= y_min:
            raise ValueError(f"y_max ({y_max}) must be > y_min ({y_min})")

    # Validate Z-axis
    if z_min is not None and z_max is not None:
        _ensure_finite("z_min", z_min)
        _ensure_finite("z_max", z_max)
        if z_max <= z_min:
            raise ValueError(f"z_max ({z_max}) must be > z_min ({z_min})")

    # 5. Mask consistency (Shape check against numpy array)
    expected_shape = (nx, ny, nz)
    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match grid counts {expected_shape}"
        )
    
    # Ensure mask contains valid entries (-1: obstacle, 0: fluid, 1: boundary)
    if not np.all(np.isin(mask, [-1, 0, 1])):
        raise ValueError("Mask contains invalid entries (only -1, 0, 1 allowed)")

    # 6. Field Finiteness (Checking the actual allocated numpy arrays)
    # This prevents the solver from starting with NaN/Inf values that crash the PPE
    for field_name in ["U", "V", "W", "P"]:
        if field_name in fields:
            arr = fields[field_name]
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Initial field '{field_name}' contains non-finite values (Inf/NaN)")