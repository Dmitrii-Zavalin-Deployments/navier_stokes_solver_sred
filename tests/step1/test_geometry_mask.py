# tests/step1/test_geometry_mask.py

import numpy as np
import pytest

from src.step1.map_geometry_mask import map_geometry_mask


def test_perfect_reshape():
    nx, ny, nz = 4, 4, 4
    flat = [1] * (nx * ny * nz)
    order = "i + nx*(j + ny*k)"

    mask = map_geometry_mask(flat, (nx, ny, nz), order)

    assert mask.shape == (4, 4, 4)
    assert mask.dtype == int


def test_length_mismatch():
    nx, ny, nz = 4, 4, 4
    flat = [1] * (nx * ny * nz - 1)
    order = "i + nx*(j + ny*k)"

    with pytest.raises(ValueError):
        map_geometry_mask(flat, (nx, ny, nz), order)


def test_data_type_pollution():
    nx, ny, nz = 2, 2, 2
    bad_flat = [1, 0, "3", 1, 0, 1, 0, 1]
    order = "i + nx*(j + ny*k)"

    # Step 1 must reject non-integer or non-finite values
    with pytest.raises(ValueError):
        map_geometry_mask(bad_flat, (nx, ny, nz), order)


def test_flattening_order_round_trip():
    nx, ny, nz = 3, 3, 3
    flat = [
        1 if (i + j + k) % 2 == 0 else 0
        for k in range(nz)
        for j in range(ny)
        for i in range(nx)
    ]
    order = "i + nx*(j + ny*k)"

    mask = map_geometry_mask(flat, (nx, ny, nz), order)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = i + nx * (j + ny * k)
                assert mask[i, j, k] == flat[idx]
