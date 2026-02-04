# tests/step1/test_geometry_mask.py
import numpy as np
import pytest

from src.step1.map_geometry_mask import map_geometry_mask


def test_perfect_reshape():
    nx, ny, nz = 4, 4, 4
    flat = list(range(nx * ny * nz))
    order = "i + nx*(j + ny*k)"

    mask = map_geometry_mask(flat, (nx, ny, nz), order)

    assert mask.shape == (4, 4, 4)
    assert mask.dtype == int


def test_length_mismatch():
    nx, ny, nz = 4, 4, 4
    flat = list(range(nx * ny * nz - 1))  # 63 instead of 64
    order = "i + nx*(j + ny*k)"

    with pytest.raises(ValueError):
        map_geometry_mask(flat, (nx, ny, nz), order)


def test_data_type_pollution():
    nx, ny, nz = 2, 2, 2
    bad_flat = [1, 2, "3", 4, 5, 6, 7, 8]
    order = "i + nx*(j + ny*k)"

    with pytest.raises(TypeError):
        map_geometry_mask(bad_flat, (nx, ny, nz), order)


def test_opaque_label_acceptance():
    nx, ny, nz = 2, 2, 2
    flat = [0, -99, 500, 3, 7, 8, 9, 10]
    order = "i + nx*(j + ny*k)"

    mask = map_geometry_mask(flat, (nx, ny, nz), order)

    assert mask.shape == (2, 2, 2)
    assert mask[0, 1, 0] == 500  # arbitrary integer accepted


def test_flattening_order_round_trip():
    nx, ny, nz = 3, 3, 3
    flat = list(range(nx * ny * nz))
    order = "i + nx*(j + ny*k)"

    mask = map_geometry_mask(flat, (nx, ny, nz), order)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = i + nx * (j + ny * k)
                assert mask[i, j, k] == flat[idx]
