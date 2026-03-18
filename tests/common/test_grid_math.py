# tests/common/test_grid_math.py

import numpy as np
import pytest

from src.common.grid_math import get_coords_from_index, get_flat_index

# Base configuration for standard tests
NX, NY, NZ = 4, 4, 4
# Buffered configuration (Halo/Ghost cells: +2 to each dimension)
BUF_NX, BUF_NY, BUF_NZ = NX + 2, NY + 2, NZ + 2

def test_manual_mapping():
    """Verify specific known values for 3D to 1D and vice-versa."""
    # (i, j, k) = (2, 2, 2) in a 4x4x4 grid
    # index = 2 + (4 * 2) + (4 * 4 * 2) = 2 + 8 + 32 = 42
    assert get_flat_index(2, 2, 2, NX, NY) == 42
    assert get_coords_from_index(42, NX, NY) == (2, 2, 2)

def test_edge_cases():
    """Test boundaries of the 3D space."""
    # Origin
    assert get_flat_index(0, 0, 0, NX, NY) == 0
    assert get_coords_from_index(0, NX, NY) == (0, 0, 0)
    
    # Last element
    max_idx = (NX * NY * NZ) - 1
    assert get_flat_index(NX-1, NY-1, NZ-1, NX, NY) == max_idx
    assert get_coords_from_index(max_idx, NX, NY) == (NX-1, NY-1, NZ-1)

def test_halo_offset_integrity():
    """
    CRITICAL: Verify the +1 offset math used for Ghost/Halo cells.
    This ensures that interior index (0,0,0) correctly maps to 
    buffered index (1,1,1) without drift.
    """
    # In a 6x6 grid (4+2), (1,1,1) should be: 1 + (6*1) + (36*1) = 43
    buf_idx = get_flat_index(1, 1, 1, BUF_NX, BUF_NY)
    assert buf_idx == 43
    
    coords = get_coords_from_index(buf_idx, BUF_NX, BUF_NY)
    assert coords == (1, 1, 1), f"Ghost cell coordinate drift: expected (1,1,1), got {coords}"

@pytest.mark.parametrize("i, j, k", [
    (0, 0, 0), (3, 3, 3), (1, 2, 3), (0, 3, 2), (2, 0, 1)
])
def test_round_trip_3d_to_1d_to_3d(i, j, k):
    """Verify that 3D coordinates survive the transition to 1D and back."""
    flat = get_flat_index(i, j, k, NX, NY)
    coords = get_coords_from_index(flat, NX, NY)
    assert coords == (i, j, k)

@pytest.mark.parametrize("index", range(0, NX * NY * NZ, 5))
def test_round_trip_1d_to_3d_to_1d(index):
    """Verify that a flat index survives the transition to 3D and back."""
    coords = get_coords_from_index(index, NX, NY)
    flat = get_flat_index(*coords, NX, NY)
    assert flat == index

def test_data_alignment_roundtrip():
    """
    Simulates placing data in a buffered foundation and retrieving it.
    Replaces the logic from the deleted test_grid_mapping_integrity.py.
    """
    total_buf_cells = BUF_NX * BUF_NY * BUF_NZ
    foundation = np.zeros(total_buf_cells)
    
    # Map a few interior points to the buffered foundation
    test_points = [(0, 0, 0), (2, 1, 3), (NX-1, NY-1, NZ-1)]
    for (i, j, k) in test_points:
        # Offset +1 for Ghost Cells
        flat_idx = get_flat_index(i + 1, j + 1, k + 1, BUF_NX, BUF_NY)
        foundation[flat_idx] = 99.9  # Marker value
        
        # Verify roundtrip
        retrieved_coords = get_coords_from_index(flat_idx, BUF_NX, BUF_NY)
        assert retrieved_coords == (i + 1, j + 1, k + 1)
        assert foundation[flat_idx] == 99.9

def test_exhaustive_range():
    """Verify that every single index in the volume maps correctly."""
    total_cells = NX * NY * NZ
    for idx in range(total_cells):
        i, j, k = get_coords_from_index(idx, NX, NY)
        assert get_flat_index(i, j, k, NX, NY) == idx