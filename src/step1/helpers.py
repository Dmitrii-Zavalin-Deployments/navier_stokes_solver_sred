# src/step1/helpers.py

import numpy as np

from src.common.solver_input import GridInput

# Rule 7: Granular Traceability
DEBUG = True

def generate_3d_masks(mask_data: list[int], grid: GridInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms flat input into 3D topology arrays and verified boolean masks.
    
    Compliance:
    - Rule 0 (Performance): NumPy used for vectorized topology representation.
    - Rule 5 (Deterministic): Strict size validation; no fallback defaults.
    """
    # Grid access via GridInput properties (SSoT)
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    
    expected_len = nx * ny * nz
    if len(mask_data) != expected_len:
        # Rule 5: Explicit or Error. Silent truncation or padding is prohibited.
        raise ValueError(f"Mask size mismatch: Expected {expected_len}, got {len(mask_data)}")

    # Fortran-style ('F') ordering maintains (i, j, k) logical indexing for spatial solvers
    # This must match the index calculation used in factory.py (i + nx * (j + ny * k))
    mask_3d = np.asarray(mask_data, dtype=np.int8).reshape((nx, ny, nz), order="F")
    
    # Logic-Layer: Identify fluid and boundary regions via vectorized masks
    is_fluid = (mask_3d == 1)
    is_boundary = (mask_3d == -1)
    
    if DEBUG:
        print(f"DEBUG [Step 1.2]: Topology Verification (Mask Generated)")
        print(f"  > Grid Dimensions: {nx}x{ny}x{nz}")
        print(f"  > Fluid Volume: {np.sum(is_fluid)} cells")
        
    return mask_3d, is_fluid, is_boundary

def parse_bc_lookup(bc_list: list) -> dict[str, dict]:
    """
    Converts BC input into a lookup table. 
    Uses None as a sentinel for missing physics, ensuring explicit data state.
    """
    table = {}
    for item in bc_list:
        # Initialize with None to signify 'not applicable'
        bc_entry = {
            "type": str(item.type),
            "u": None, "v": None, "w": None, "p": None
        }
        
        # Explicit access to item values
        # We enforce strict floating point conversion per Rule 5
        for key in ["u", "v", "w", "p"]:
            if hasattr(item.values, key) and getattr(item.values, key) is not None:
                bc_entry[key] = float(getattr(item.values, key))
        
        table[str(item.location)] = bc_entry
        
        if DEBUG:
            print(f"DEBUG [Step 1.3]: BC Map Entry Created -> Location: {item.location}, Type: {item.type}")
            
    return table