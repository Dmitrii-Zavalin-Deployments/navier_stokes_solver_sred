# src/step1/helpers.py

import numpy as np
from typing import Dict, List, Tuple
from src.solver_input import GridInput, BoundaryConditionItem

def allocate_staggered_fields(grid: GridInput) -> Dict[str, np.ndarray]:
    """Allocates memory for the Harlow-Welch staggered grid."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    return {
        "P": np.zeros((nx, ny, nz), dtype=np.float64),
        "U": np.zeros((nx + 1, ny, nz), dtype=np.float64),
        "V": np.zeros((nx, ny + 1, nz), dtype=np.float64),
        "W": np.zeros((nx, ny, nz + 1), dtype=np.float64),
    }

def generate_3d_masks(mask_data: List[int], grid: GridInput) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transforms the flat JSON mask into 3D topology arrays."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    mask_3d = np.asarray(mask_data, dtype=np.int8).reshape((nx, ny, nz), order="F")
    
    is_fluid = (mask_3d == 1)
    is_boundary = (mask_3d == -1)
    return mask_3d, is_fluid, is_boundary

def parse_bc_lookup(items: List[BoundaryConditionItem]) -> Dict[str, Dict]:
    """Converts BC list into a high-speed lookup table for physics kernels."""
    table = {}
    for item in items:
        table[item.location] = {
            "type": item.type,
            "u": float(item.values.get("u", 0.0)),
            "v": float(item.values.get("v", 0.0)),
            "w": float(item.values.get("w", 0.0)),
            "p": float(item.values.get("p", 0.0))
        }
    return table