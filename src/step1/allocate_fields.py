# src/step1/allocate_fields.py

from __future__ import annotations
from src.solver_input import GridInput
from typing import Dict, Any
import numpy as np

def allocate_fields(grid: GridInput) -> Dict[str, np.ndarray]:
    """
    Allocates cell-centered pressure and staggered velocity fields.
    
    Constitutional Role: Memory Architect.
    Logic: Face-centered staggered discretization (Harlow-Welch).
    
    Args:
        grid: Dictionary containing 'nx', 'ny', 'nz'.
        
    Returns:
        Dict[str, np.ndarray]: Initialized fields for P, U, V, and W.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # Integrity Check: Prevent zero or negative allocation
    if any(dim <= 0 for dim in (nx, ny, nz)):
        raise ValueError(f"Invalid grid dimensions for allocation: {nx}x{ny}x{nz}")

    # P: Cell-centered (Nx, Ny, Nz)
    # U, V, W: Face-centered (Staggered by +1 in their respective directions)
    return {
        "P": np.zeros((nx, ny, nz), dtype=np.float64),
        "U": np.zeros((nx + 1, ny, nz), dtype=np.float64),
        "V": np.zeros((nx, ny + 1, nz), dtype=np.float64),
        "W": np.zeros((nx, ny, nz + 1), dtype=np.float64),
    }