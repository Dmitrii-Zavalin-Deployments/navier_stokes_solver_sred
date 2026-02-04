# file: src/step1/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np


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
        # Enforce positive integer grid counts
        for name, val in [("nx", self.nx), ("ny", self.ny), ("nz", self.nz)]:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val}")


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


# ------------------------------------------------------------
# Final assembled Step 1 state
# ------------------------------------------------------------
@dataclass
class SimulationState:
    config: Config
    grid: GridConfig
    fields: Fields
    mask_3d: np.ndarray
    boundary_table: Dict[str, List[Dict[str, Any]]]
    constants: DerivedConstants
