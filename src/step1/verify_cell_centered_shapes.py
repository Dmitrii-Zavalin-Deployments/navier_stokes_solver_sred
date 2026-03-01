# src/step1/verify_cell_centered_shapes.py

from __future__ import annotations

from typing import Dict, Any
import numpy as np

def verify_cell_centered_shapes(state: Dict[str, Any]) -> None:
    """
    Ensures all ingested arrays match the basic cell-centered grid dimensions.
    
    Constitutional Role: Staging Area Guard.
    Requirement: Converts JSON-parsed lists to NumPy arrays in-place.
    
    Args:
        state: The Step 1 output dictionary containing 'grid' and 'fields'.
    """

    # 1. Structural Guard: Skip if grid is missing (Negative Schema Tests)
    if "grid" not in state:
        return

    grid = state["grid"]
    fields = state["fields"]

    try:
        nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    except (KeyError, TypeError, ValueError):
        # Grid exists but is malformed; validation fails elsewhere
        return

    expected_shape = (nx, ny, nz)

    # 2. Type Conversion & Normalization (List -> ndarray)
    # We enforce specific dtypes here to prevent downstream precision issues.
    mapping = {
        "P": np.float64,
        "U": np.float64,
        "V": np.float64,
        "W": np.float64,
        "Mask": np.int8
    }

    for name, dtype in mapping.items():
        if name not in fields:
            raise KeyError(f"Missing essential field for verification: {name}")
        
        val = fields[name]
        
        # Convert list or tuple to numpy array
        if isinstance(val, (list, tuple)):
            fields[name] = np.asarray(val, dtype=dtype)
        elif not isinstance(val, np.ndarray):
            raise TypeError(f"Field '{name}' must be a list or numpy array, got {type(val)}")
        
        # 3. Shape Verification
        actual_shape = fields[name].shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"Dimension Mismatch on '{name}': "
                f"Expected {expected_shape} (nx, ny, nz), but got {actual_shape}."
            )