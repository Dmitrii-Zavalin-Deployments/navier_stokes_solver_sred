# src/step1/initialize_grid.py

from __future__ import annotations
from typing import Dict, Any
import math

def initialize_grid(grid: Dict[str, Any]) -> Dict[str, Any]:
    """
    Grid initializer with physically meaningful spacing.
    Calculates Delta values from domain extents.
    """

    # 1. Extraction with Type Enforcement
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    x_min, x_max = float(grid["x_min"]), float(grid["x_max"])
    y_min, y_max = float(grid["y_min"]), float(grid["y_max"])
    z_min, z_max = float(grid["z_min"]), float(grid["z_max"])

    # 2. Domain Integrity Validation
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("Grid dimensions (nx, ny, nz) must be positive integers.")

    for label, val in [("x", (x_min, x_max)), ("y", (y_min, y_max)), ("z", (z_min, z_max))]:
        if not (math.isfinite(val[0]) and math.isfinite(val[1])):
            raise ValueError(f"{label} extents must be finite numbers.")
        if val[1] <= val[0]:
            raise ValueError(f"{label}_max must be strictly greater than {label}_min.")

    # 3. Compute Physical Spacing (The Math Gate)
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    # 4. Spacing Safety Check
    for dim, spacing in [("dx", dx), ("dy", dy), ("dz", dz)]:
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError(f"Computed {dim} is invalid ({spacing}). Check extents vs dimensions.")

    # 5. Return Full Grid Context
    # Including min/max here ensures traceability in the SolverState.grid attribute
    return {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": dx, "dy": dy, "dz": dz,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max
    }