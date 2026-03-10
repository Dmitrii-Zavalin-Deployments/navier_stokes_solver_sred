# src/step1/helpers.py

import numpy as np

from src.solver_input import BoundaryConditionItem, GridInput

# Rule 7: Granular Traceability
DEBUG = True

def generate_3d_masks(mask_data: list[int], grid: GridInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms flat input into 3D topology arrays.
    
    Compliance:
    - Rule 0 (Performance): NumPy used for vectorized topology representation.
    - Rule 5 (Deterministic): Strict size validation; no fallback defaults.
    """
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    
    expected_len = nx * ny * nz
    if len(mask_data) != expected_len:
        # Rule 5: Explicit or Error. Silent truncation or padding is prohibited.
        raise ValueError(f"Mask size mismatch: Expected {expected_len}, got {len(mask_data)}")

    # Fortran-style ('F') ordering maintains (i, j, k) logical indexing
    mask_3d = np.asarray(mask_data, dtype=np.int8).reshape((nx, ny, nz), order="F")
    
    # Logic-Layer: Identify fluid and boundary regions via vectorized masks
    is_fluid = (mask_3d == 1)
    is_boundary = (mask_3d == -1)
    
    if DEBUG:
        print(f"DEBUG [Step 1.2]: Topology Verification (Mask Generated)")
        print(f"  > Target Domain: {nx}x{ny}x{nz}")
        print(f"  > Fluid Volume: {np.sum(is_fluid)} cells")
        
    return mask_3d, is_fluid, is_boundary

def parse_bc_lookup(items: list[BoundaryConditionItem]) -> dict[str, dict]:
    """
    Converts BC list into a high-speed lookup table for the BoundaryManager.
    
    Compliance:
    - Rule 8 (Singular Access): Primary interface for boundary definition.
    - Rule 5 (Deterministic): Direct access to dictionary keys ensures 
      missing physics data triggers an immediate KeyError.
    """
    table = {}
    for item in items:
        # Direct key access ensures that any missing physical parameters 
        # (u, v, w, p) results in an immediate crash, adhering to the Zero-Debt mandate.
        table[str(item.location)] = {
            "type": str(item.type),
            "u": float(item.values["u"]),
            "v": float(item.values["v"]),
            "w": float(item.values["w"]),
            "p": float(item.values["p"])
        }
        if DEBUG:
            print(f"DEBUG [Step 1.3]: BC Map Entry Created -> Location: {item.location}, Type: {item.type}")
            
    return table