# tests/step1/test_map_geometry_mask_full.py

import numpy as np
import pytest
from src.step1.map_geometry_mask import map_geometry_mask
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_input():
    """Provides the canonical dummy input (Section 5 Compliance)."""
    return solver_input_schema_dummy()

def test_map_geometry_mask_order_integrity(dummy_input):
    """
    Verifies that the flat mask list is reshaped into a 3D array
    matching the solver's coordinate convention (i, j, k) using
    Fortran order (i changes fastest).
    """
    grid = dummy_input["grid"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    total_cells = nx * ny * nz

    # Probes: Index 0 (First), 1 (X-Step), nx (Y-Step), 
    # nx*ny-1 (Top-Right Front), total_cells-1 (Very Last)
    probes = [0, 1, nx, nx * ny - 1, total_cells - 1]

    for target_index in probes:
        # Create a valid flat mask (all fluid = 0)
        flat = [0] * total_cells
        # Place a single obstacle (-1) at the target index
        flat[target_index] = -1
        
        # Act
        mask_3d, _, _ = map_geometry_mask(flat, grid)

        # Fortran unravelling logic:
        # k = layer (slowest), j = row, i = column (fastest)
        k_exp = target_index // (nx * ny)
        remainder = target_index % (nx * ny)
        j_exp = remainder // nx
        i_exp = remainder % nx

        # Assert: Exact spatial positioning
        assert mask_3d[i_exp, j_exp, k_exp] == -1, (
            f"Spatial audit failed: Flat index {target_index} mapped to "
            f"({mask_3d.shape}) at wrong coordinates."
        )

def test_map_geometry_mask_invalid_values(dummy_input):
    """Ensures values outside [-1, 0, 1] are rejected (Domain Validation)."""
    grid = dummy_input["grid"]
    flat = [0] * (grid["nx"] * grid["ny"] * grid["nz"])
    
    # Value 5 is outside the physical definition range
    flat[0] = 5  
    
    with pytest.raises(ValueError, match="(?i)entries must be -1, 0, or 1"):
        map_geometry_mask(flat, grid)