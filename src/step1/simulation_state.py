from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class Grid:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float


@dataclass
class SimulationState:
    grid: Grid
    P: np.ndarray
    U: np.ndarray
    V: np.ndarray
    W: np.ndarray
    mask: np.ndarray
    constants: Dict[str, float]
    config: Dict[str, Any]
