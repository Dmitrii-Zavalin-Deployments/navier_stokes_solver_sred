# file: step1/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


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


@dataclass
class Config:
    domain: Dict[str, Any]
    fluid: Dict[str, Any]
    simulation: Dict[str, Any]
    forces: Dict[str, Any]
    boundary_conditions: List[Dict[str, Any]]
    geometry_definition: Dict[str, Any]


@dataclass
class Fields:
    P: np.ndarray
    U: np.ndarray
    V: np.ndarray
    W: np.ndarray
    Mask: np.ndarray


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


@dataclass
class SimulationState:
    config: Config
    grid: GridConfig
    fields: Fields
    mask_3d: np.ndarray
    boundary_table: Dict[str, List[Dict[str, Any]]]
    constants: DerivedConstants
