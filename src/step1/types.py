# file: src/step1/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import math


# ------------------------------------------------------------
# Grid configuration (domain + discretization)
# ------------------------------------------------------------
@dataclass
class GridConfig:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    def __post_init__(self):
        # Validate integer grid counts
        for name, val in [("nx", self.nx), ("ny", self.ny), ("nz", self.nz)]:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val}")

        # Validate finite positive spacings
        for name, val in [("dx", self.dx), ("dy", self.dy), ("dz", self.dz)]:
            if not (isinstance(val, (int, float)) and math.isfinite(val) and val > 0):
                raise ValueError(f"{name} must be a finite positive number, got {val}")

        # Validate finite extents
        for name, val in [
            ("x_min", self.x_min), ("x_max", self.x_max),
            ("y_min", self.y_min), ("y_max", self.y_max),
            ("z_min", self.z_min), ("z_max", self.z_max),
        ]:
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                raise ValueError(f"{name} must be a finite number, got {val}")


# ------------------------------------------------------------
# Staggered MAC-grid fields
# ------------------------------------------------------------
@dataclass
class Fields:
    P: np.ndarray   # (nx, ny, nz)
    U: np.ndarray   # (nx+1, ny, nz)
    V: np.ndarray   # (nx, ny+1, nz)
    W: np.ndarray   # (nx, ny, nz+1)
    Mask: np.ndarray  # (nx, ny, nz)

    def __post_init__(self):
        # Validate array types
        for name, arr in [
            ("P", self.P), ("U", self.U),
            ("V", self.V), ("W", self.W),
            ("Mask", self.Mask),
        ]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy array")

        # Validate mask values
        if not np.isin(self.Mask, [-1, 0, 1]).all():
            raise ValueError("Mask must contain only -1, 0, or 1")


# ------------------------------------------------------------
# Parsed configuration from JSON
# ------------------------------------------------------------
@dataclass
class Config:
    domain: Dict[str, Any]
    fluid: Dict[str, Any]
    simulation: Dict[str, Any]
    forces: Dict[str, Any]
    boundary_conditions: List[Dict[str, Any]]
    geometry_definition: Dict[str, Any]


# ------------------------------------------------------------
# Precomputed numerical constants
# ------------------------------------------------------------
@dataclass
class DerivedConstants:
    rho: float
    mu: float
    dt: float
    dx: float
    dy: float
    dz: float
    inv_dx: float
    inv_dy: float
    inv_dz: float
    inv_dx2: float
    inv_dy2: float
    inv_dz2: float

    def __post_init__(self):
        # All constants must be finite
        for name, val in self.__dict__.items():
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                raise ValueError(f"{name} must be a finite number, got {val}")

        # Positivity constraints
        if self.rho <= 0:
            raise ValueError("rho must be positive")
        if self.mu < 0:
            raise ValueError("mu must be non-negative")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        for name in ["dx", "dy", "dz"]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


# ------------------------------------------------------------
# Final assembled Step 1 state (internal only)
# ------------------------------------------------------------
@dataclass
class SimulationState:
    config: Config
    grid: GridConfig
    fields: Fields
    mask_3d: np.ndarray
    boundary_table: Dict[str, List[Dict[str, Any]]]
    constants: DerivedConstants

    def __post_init__(self):
        if not isinstance(self.mask_3d, np.ndarray):
            raise TypeError("mask_3d must be a numpy array")
