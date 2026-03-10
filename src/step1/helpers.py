# src/step1/helpers.py

import numpy as np

from src.solver_input import BoundaryConditionItem, GridInput

# Rule 7: Granular Traceability
DEBUG = True

def generate_3d_masks(mask_data: list[int], grid: GridInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms the flat JSON mask into 3D topology arrays.
    Theory Mapping: Section 6 - Mask-Based Geometry.
    """
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    
    expected_len = nx * ny * nz
    if len(mask_data) != expected_len:
        # Rule 5: Explicit or Error - No fallback sizes
        raise ValueError(f"Mask size mismatch: Expected {expected_len}, got {len(mask_data)}")

    # Order 'F' is critical for Fortran-style indexing (i, j, k)
    mask_3d = np.asarray(mask_data, dtype=np.int8).reshape((nx, ny, nz), order="F")
    
    is_fluid = (mask_3d == 1)
    is_boundary = (mask_3d == -1)
    
    if DEBUG:
        print(f"DEBUG [Step 1.2]: Topology Verification:")
        print(f"  > Target Domain: {nx}x{ny}x{nz}")
        print(f"  > Fluid Volume: {np.sum(is_fluid)} cells")
        
    return mask_3d, is_fluid, is_boundary

def parse_bc_lookup(items: list[BoundaryConditionItem]) -> dict[str, dict]:
    """
    Converts BC list into a high-speed lookup table.
    Rule 5 Violation Fixed: Removed all .get() defaults.
    """
    table = {}
    for item in items:
        # Accessing keys directly: If values are missing, this raises KeyError immediately.
        # This complies with the "Zero-Debt" and "Explicit or Error" mandates.
        table[item.location] = {
            "type": str(item.type),
            "u": float(item.values["u"]),
            "v": float(item.values["v"]),
            "w": float(item.values["w"]),
            "p": float(item.values["p"])
        }
        if DEBUG:
            print(f"DEBUG [Step 1.3]: BC Map -> {item.location}: type={item.type}")
            
    return table