# src/step1/validate_physical_constraints.py

from __future__ import annotations
from typing import TYPE_CHECKING
import math
import numpy as np

if TYPE_CHECKING:
    from src.solver_state import SolverState

def _ensure_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Stability Violation: {name} must be finite and > 0, got {value}")

def _ensure_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"Physicality Violation: {name} must be finite and >= 0, got {value}")

def _ensure_positive_int(name: str, value: int) -> None:
    if not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"Topology Violation: {name} must be a positive integer, got {value}")

def _ensure_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"Precision Error: {name} must be finite, got {value}")

def validate_physical_constraints(state: SolverState) -> None:
    """
    Final integrity gate for Step 1.
    Ensures all state attributes are mathematically and physically coherent.
    
    Constitutional Role: Logical Firewall.
    Audit: Phase F Data Completeness.
    """
    grid = state.grid
    constants = state.constants
    fields = state.fields
    
    # SURGICAL FIX: Accessing mask via the 'masks' department
    mask = state.masks.mask

    # 1. Structural Integrity Guard
    required_grid_keys = ["nx", "ny", "nz", "dx", "dy", "dz"]
    missing_keys = [k for k in required_grid_keys if k not in grid]
    if missing_keys:
        raise ValueError(f"Incomplete Grid Definition: Missing keys {missing_keys}")

    # 2. Fluid & Temporal Constraints
    _ensure_positive("density (rho)", float(constants["rho"]))
    _ensure_non_negative("viscosity (mu)", float(constants["mu"]))
    _ensure_positive("time step (dt)", float(constants["dt"]))

    # 3. Spatial Resolution Constraints
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    _ensure_positive_int("nx", nx)
    _ensure_positive_int("ny", ny)
    _ensure_positive_int("nz", nz)
    
    _ensure_positive("dx", float(grid["dx"]))
    _ensure_positive("dy", float(grid["dy"]))
    _ensure_positive("dz", float(grid["dz"]))

    # 4. Domain Extent Validation (Directional Integrity)
    for dim in ['x', 'y', 'z']:
        v_min, v_max = grid.get(f"{dim}_min"), grid.get(f"{dim}_max")
        if v_min is not None and v_max is not None:
            _ensure_finite(f"{dim}_min", v_min)
            _ensure_finite(f"{dim}_max", v_max)
            if v_max <= v_min:
                raise ValueError(f"Domain Inversion: {dim}_max ({v_max}) <= {dim}_min ({v_min})")

    # 5. Topological Consistency
    # CONSTITUTIONAL FIX: Mask is accessed from the MaskData container.
    if mask is not None:
        expected_len = nx * ny * nz
        if len(mask) != expected_len:
            raise ValueError(f"Mask Length Mismatch: {len(mask)} != {expected_len}")
        
        # Check values
        unauthorized = [v for v in mask if v not in {-1, 0, 1}]
        if unauthorized:
            raise ValueError(f"Forbidden Topology: Mask contains values outside range {-1, 0, 1}: {set(unauthorized)}")

    # 6. Field Sanity Check (NaN/Inf Propagation Prevention)
    for field_name in ["U", "V", "W", "P"]:
        if field_name in fields:
            if not np.all(np.isfinite(fields[field_name])):
                raise ValueError(f"Genesis Error: Field '{field_name}' contains non-finite values.")
