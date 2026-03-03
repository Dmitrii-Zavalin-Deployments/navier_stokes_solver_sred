# src/step1/helpers.py

import numpy as np
from typing import Dict, List, Tuple
from src.solver_input import GridInput, BoundaryConditionItem

# Global Debug Toggle
DEBUG = True

def allocate_staggered_fields(grid: GridInput) -> Dict[str, np.ndarray]:
    """Allocates memory for the Harlow-Welch staggered grid."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    fields = {
        "P": np.zeros((nx, ny, nz), dtype=np.float64),
        "U": np.zeros((nx + 1, ny, nz), dtype=np.float64),
        "V": np.zeros((nx, ny + 1, nz), dtype=np.float64),
        "W": np.zeros((nx, ny, nz + 1), dtype=np.float64),
    }

    if DEBUG:
        for name, arr in fields.items():
            print(f"DEBUG [Step 1]: Field '{name}' allocated with shape {arr.shape}")

    return fields

def generate_3d_masks(mask_data: List[int], grid: GridInput) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transforms the flat JSON mask into 3D topology arrays."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    if DEBUG:
        print(f"DEBUG [Step 1]: Reshaping mask data of length {len(mask_data)}")

    mask_3d = np.asarray(mask_data, dtype=np.int8).reshape((nx, ny, nz), order="F")
    
    if DEBUG:
        print(f"DEBUG [Step 1]: Mask 3D reshaped with order='F'. Unique values: {np.unique(mask_3d)}")

    is_fluid = (mask_3d == 1)
    is_boundary = (mask_3d == -1)
    
    if DEBUG:
        print(f"DEBUG [Step 1]: Fluid cells: {np.sum(is_fluid)}, Boundary cells: {np.sum(is_boundary)}")
        
    return mask_3d, is_fluid, is_boundary

def parse_bc_lookup(items: List[BoundaryConditionItem]) -> Dict[str, Dict]:
    """Converts BC list into a high-speed lookup table for physics kernels."""
    if DEBUG:
        print(f"DEBUG [Step 1]: Parsing {len(items)} boundary condition items")

    table = {}
    for item in items:
        table[item.location] = {
            "type": item.type,
            "u": float(item.values.get("u", 0.0)),
            "v": float(item.values.get("v", 0.0)),
            "w": float(item.values.get("w", 0.0)),
            "p": float(item.values.get("p", 0.0))
        }
        if DEBUG:
            print(f"DEBUG [Step 1]: BC at {item.location} set to {item.type} with p={table[item.location]['p']}")
            
    return table