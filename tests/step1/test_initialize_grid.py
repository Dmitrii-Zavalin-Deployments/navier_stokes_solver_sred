# tests/step1/test_initialize_grid.py

import pytest
from src.step1.initialize_grid import initialize_grid


def test_initialize_grid_basic():
    domain = {
        "nx": 4, "ny": 6, "nz": 2,
        "x_min": 0.0, "x_max": 4.0,
        "y_min": 0.0, "y_max": 6.0,
        "z_min": 0.0, "z_max": 2.0,
    }

    grid = initialize_grid(domain)

    assert grid.nx == 4
    assert grid.dx == 1.0
    assert grid.dy == 1.0
    assert grid.dz == 1.0


def test_initialize_grid_invalid_extents():
    domain = {
        "nx": 4, "ny": 4, "nz": 4,
        "x_min": 1.0, "x_max": 1.0,  # invalid: zero extent
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0,
    }

    with pytest.raises(ValueError):
        initialize_grid(domain)
