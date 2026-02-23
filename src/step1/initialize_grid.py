# src/step1/initialize_grid.py

from __future__ import annotations
from typing import Dict, Any
import math

def initialize_grid(grid_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes grid spacing and validates domain extents.
    
    Constitutional Role: Spatial Governor.
    Logic: Finite Difference Uniform Grid Discretization.
    
    Returns:
        Dict[str, Any]: Complete grid context including delta values.
    """

    # 1. Extraction with Strict Type Enforcement
    # Values from JSON schema are cast to ensure mathematical consistency.
    try:
        nx, ny, nz = int(grid_input["nx"]), int(grid_input["ny"]), int(grid_input["nz"])
        x_min, x_max = float(grid_input["x_min"]), float(grid_input["x_max"])
        y_min, y_max = float(grid_input["y_min"]), float(grid_input["y_max"])
        z_min, z_max = float(grid_input["z_min"]), float(grid_input["z_max"])
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Grid initialization failed due to missing or invalid types: {e}")

    # 2. Domain Integrity Validation (Phase A.1)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Grid dimensions must be positive. Got: {nx}, {ny}, {nz}")

    for label, min_val, max_val in [("x", x_min, x_max), ("y", y_min, y_max), ("z", z_min, z_max)]:
        if not (math.isfinite(min_val) and math.isfinite(max_val)):
            raise ValueError(f"Domain extents for {label} must be finite.")
        if max_val <= min_val:
            raise ValueError(f"Inverted domain: {label}_max ({max_val}) must be > {label}_min ({min_val}).")

    # 3. Compute Physical Spacing (The Math Gate)
    # dx = L / N
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    # 4. Return Full Grid Context
    # This dictionary is injected into SolverState.config['grid'] to maintain symmetry.
    return {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": dx, "dy": dy, "dz": dz,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max
    }