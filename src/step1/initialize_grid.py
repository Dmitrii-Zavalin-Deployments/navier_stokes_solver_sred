# src/step1/initialize_grid.py

from __future__ import annotations
from typing import Dict, Any
import math


def initialize_grid(grid: Dict[str, Any]) -> Dict[str, Any]:
    """
    Grid initializer with physically meaningful spacing.

    Computes:
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny
        dz = (z_max - z_min) / nz

    Step 1 validates:
        • grid dimensions > 0
        • extents finite
        • extents ordered
        • computed spacing finite and positive
    """

    # ---------------------------------------------------------
    # Extract values
    # ---------------------------------------------------------
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    x_min = float(grid["x_min"])
    x_max = float(grid["x_max"])
    y_min = float(grid["y_min"])
    y_max = float(grid["y_max"])
    z_min = float(grid["z_min"])
    z_max = float(grid["z_max"])

    # ---------------------------------------------------------
    # Validate grid dimensions
    # ---------------------------------------------------------
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("Grid dimensions nx, ny, nz must be positive.")

    # ---------------------------------------------------------
    # Validate finiteness of extents
    # ---------------------------------------------------------
    for value in (x_min, x_max, y_min, y_max, z_min, z_max):
        if not math.isfinite(value):
            raise ValueError("Grid extents must be finite.")

    # ---------------------------------------------------------
    # Validate ordering of extents
    # ---------------------------------------------------------
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")
    if y_max <= y_min:
        raise ValueError("y_max must be greater than y_min.")
    if z_max <= z_min:
        raise ValueError("z_max must be greater than z_min.")

    # ---------------------------------------------------------
    # Compute physical spacing
    # ---------------------------------------------------------
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    # ---------------------------------------------------------
    # Validate spacing
    # ---------------------------------------------------------
    for v in (dx, dy, dz):
        if not math.isfinite(v) or v <= 0:
            raise ValueError("Computed grid spacing must be positive and finite.")

    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }
