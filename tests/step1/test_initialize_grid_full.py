# tests/step1/test_initialize_grid_full.py

import pytest
from src.step1.initialize_grid import initialize_grid


def test_invalid_y_extent():
    domain = {
        "nx": 2, "ny": 2, "nz": 2,
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 1.0, "y_max": 1.0,  # zero extent → invalid
        "z_min": 0.0, "z_max": 1.0,
    }

    with pytest.raises(ValueError):
        initialize_grid(domain)


def test_invalid_z_extent():
    domain = {
        "nx": 2, "ny": 2, "nz": 2,
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 2.0, "z_max": 2.0,  # zero extent → invalid
    }

    with pytest.raises(ValueError):
        initialize_grid(domain)
