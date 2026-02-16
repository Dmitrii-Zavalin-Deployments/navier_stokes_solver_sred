# src/step1/initialize_grid.py
from __future__ import annotations

import math
from .types import GridConfig


def initialize_grid(domain: dict) -> GridConfig:
    """
    Initialize grid metadata for the cell-centered solver.

    Responsibilities:
      • validate domain extents and resolution
      • compute dx, dy, dz for a uniform Cartesian grid
      • ensure all values are finite and physically meaningful

    This function does not allocate fields, interpret geometry,
    or apply boundary conditions. It only constructs GridConfig.
    """

    # -----------------------------
    # Required keys
    # -----------------------------
    required_keys = [
        "nx", "ny", "nz",
        "x_min", "x_max",
        "y_min", "y_max",
        "z_min", "z_max",
    ]

    for key in required_keys:
        if key not in domain:
            raise KeyError(f"Missing required domain key: {key}")

    # -----------------------------
    # Grid resolution (must be >= 1)
    # -----------------------------
    nx = int(domain["nx"])
    ny = int(domain["ny"])
    nz = int(domain["nz"])

    for name, val in [("nx", nx), ("ny", ny), ("nz", nz)]:
        if val < 1:
            raise ValueError(f"{name} must be >= 1, got {val}")

    # -----------------------------
    # Domain extents (finite, ordered)
    # -----------------------------
    x_min = float(domain["x_min"])
    x_max = float(domain["x_max"])
    y_min = float(domain["y_min"])
    y_max = float(domain["y_max"])
    z_min = float(domain["z_min"])
    z_max = float(domain["z_max"])

    for name, val in [
        ("x_min", x_min), ("x_max", x_max),
        ("y_min", y_min), ("y_max", y_max),
        ("z_min", z_min), ("z_max", z_max),
    ]:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val}")

    if x_max <= x_min:
        raise ValueError(f"x_max must be greater than x_min, got {x_min} .. {x_max}")
    if y_max <= y_min:
        raise ValueError(f"y_max must be greater than y_min, got {y_min} .. {y_max}")
    if z_max <= z_min:
        raise ValueError(f"z_max must be greater than z_min, got {z_min} .. {z_max}")

    # -----------------------------
    # Uniform cell-centered spacing
    # -----------------------------
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    for name, val in [("dx", dx), ("dy", dy), ("dz", dz)]:
        if not (math.isfinite(val) and val > 0):
            raise ValueError(f"{name} must be a finite positive number, got {val}")

    # -----------------------------
    # Construct GridConfig
    # -----------------------------
    return GridConfig(
        nx=nx, ny=ny, nz=nz,
        dx=dx, dy=dy, dz=dz,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_min, z_max=z_max,
    )
