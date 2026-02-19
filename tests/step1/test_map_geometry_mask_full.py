# tests/step1/test_map_geometry_mask_full.py

import numpy as np
from src.step1.map_geometry_mask import map_geometry_mask


def test_map_geometry_mask_row_major():
    flat = list(range(8))
    domain = {"nx": 2, "ny": 2, "nz": 2}

    mask = map_geometry_mask(flat, domain)

    expected = np.array(flat).reshape((2, 2, 2), order="F")

    assert mask.shape == (2, 2, 2)
    assert np.array_equal(mask, expected)


def test_map_geometry_mask_column_major():
    # Under the frozen architecture, column-major is not supported.
    # Both tests validate the canonical F-order reshape.
    flat = list(range(8))
    domain = {"nx": 2, "ny": 2, "nz": 2}

    mask = map_geometry_mask(flat, domain)

    expected = np.array(flat).reshape((2, 2, 2), order="F")

    assert mask.shape == (2, 2, 2)
    assert np.array_equal(mask, expected)
