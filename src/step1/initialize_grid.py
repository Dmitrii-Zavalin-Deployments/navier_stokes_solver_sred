# file: step1/initialize_grid.py
from __future__ import annotations

from typing import Dict

from .types import GridConfig


def initialize_grid(domain_params: Dict[str, float]) -> GridConfig:
    """
    Compute spatial discretization and grid metadata.
    """
    x_min = float(domain_params["x_min"])
    x_max = float(domain_params["x_max"])
    y_min = float(domain_params["y_min"])
    y_max = float(domain_params["y_max"])
    z_min = float(domain_params["z_min"])
    z_max = float(domain_params["z_max"])

    nx = int(domain_params["nx"])
    ny = int(domain_params["ny"])
    nz = int(domain_params["nz"])

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    return GridConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        dz=dz,
        x_min=x_min,
        y_min=y_min,
        z_min=z_min,
        x_max=x_max,
        y_max=y_max,
        z_max=z_max,
    )
