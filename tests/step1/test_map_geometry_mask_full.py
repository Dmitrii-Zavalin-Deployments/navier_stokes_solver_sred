# tests/step1/test_map_geometry_mask_full.py

import numpy as np
from src.step1.map_geometry_mask import map_geometry_mask


def test_map_geometry_mask_row_major():
    flat = list(range(8))
    shape = (2, 2, 2)

    mask = map_geometry_mask(flat, shape, "i + nx*(j + ny*k)")

    assert mask.shape == (2, 2, 2)
    assert mask[0, 0, 0] == 0
    assert mask[1, 0, 0] == 1


def test_map_geometry_mask_column_major():
    flat = list(range(8))
    shape = (2, 2, 2)

    mask = map_geometry_mask(flat, shape, "k + nz*(j + ny*i)")

    assert mask.shape == (2, 2, 2)
    assert mask[0, 0, 0] == 0
    assert mask[0, 0, 1] == 1
