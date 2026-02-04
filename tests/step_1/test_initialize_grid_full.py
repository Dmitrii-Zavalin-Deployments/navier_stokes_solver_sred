# tests/step_1/test_initialize_grid_full.py

import pytest
from src.step1.initialize_grid import initialize_grid


def test_invalid_y_extent():
    domain = {
        "nx": 2, "ny": 2, "nz": 2,
        "x_min": 0, "x_max": 1,
        "y_min": 1, "y_max": 1,
        "z_min": 0, "z_max": 1,
    }
    with pytest.raises(ValueError):
        initialize_grid(domain)


def test_invalid_z_extent():
    domain = {
        "nx": 2, "ny": 2, "nz": 2,
        "x_min": 0, "x_max": 1,
        "y_min": 0, "y_max": 1,
        "z_min": 2, "z_max": 2,
    }
    with pytest.raises(ValueError):
        initialize_grid(domain)
