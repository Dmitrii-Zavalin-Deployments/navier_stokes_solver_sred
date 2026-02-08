# file: src/step1/initialize_grid.py
from __future__ import annotations

import math
from .types import GridConfig


def initialize_grid(domain: dict) -> GridConfig:
    """
    Initialize grid metadata from validated Step 1 input.

    Enforces all constraints required by the Step 1 Output Schema:
    - nx, ny, nz >= 1
    - x_max > x_min, etc.
    - dx, dy, dz > 0
    - all values finite
    """

    # -----------------------------
    # Validate required keys
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
    # Extract and validate integer dimensions
    # -----------------------------
    nx = int(domain["nx"])
    ny = int(domain["ny"])
    nz = int(domain["nz"])

    for name, val in [("nx", nx), ("ny", ny), ("nz", nz)]:
        if val < 1:
            raise ValueError(f"{name} must be >= 1, got {val}")

    # -----------------------------
    # Extract and validate extents
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

    # -----------------------------
    # Validate extents ordering
    # -----------------------------
    if x_max <= x_min:
        raise ValueError(f"x_max must be greater than x_min, got {x_min} .. {x_max}")
    if y_max <= y_min:
        raise ValueError(f"y_max must be greater than y_min, got {y_min} .. {y_max}")
    if z_max <= z_min:
        raise ValueError(f"z_max must be greater than z_min, got {z_min} .. {z_max}")

    # -----------------------------
    # Compute spacings
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
