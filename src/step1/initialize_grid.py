# file: src/step1/initialize_grid.py
from __future__ import annotations

from .types import GridConfig


def initialize_grid(domain: dict) -> GridConfig:
    nx = int(domain["nx"])
    ny = int(domain["ny"])
    nz = int(domain["nz"])

    x_min = float(domain["x_min"])
    x_max = float(domain["x_max"])
    y_min = float(domain["y_min"])
    y_max = float(domain["y_max"])
    z_min = float(domain["z_min"])
    z_max = float(domain["z_max"])

    # NEW: validate extents
    if x_max <= x_min:
        raise ValueError(f"x_max must be greater than x_min, got {x_min} .. {x_max}")
    if y_max <= y_min:
        raise ValueError(f"y_max must be greater than y_min, got {y_min} .. {y_max}")
    if z_max <= z_min:
        raise ValueError(f"z_max must be greater than z_min, got {z_min} .. {z_max}")

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    return GridConfig(
        nx=nx, ny=ny, nz=nz,
        dx=dx, dy=dy, dz=dz,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_min, z_max=z_max,
    )
