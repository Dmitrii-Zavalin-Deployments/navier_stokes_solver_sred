# tests/step1/test_initialize_grid_math.py

import math
import pytest

from src.step1.initialize_grid import initialize_grid
from src.solver_state import GridConfig


def test_correct_grid_spacing_computation():
    domain = {
        "nx": 4, "ny": 2, "nz": 1,
        "x_min": 0.0, "x_max": 8.0,
        "y_min": -1.0, "y_max": 3.0,
        "z_min": 2.0, "z_max": 6.0,
    }

    grid = initialize_grid(domain)

    assert grid.dx == pytest.approx((8.0 - 0.0) / 4)
    assert grid.dy == pytest.approx((3.0 - (-1.0)) / 2)
    assert grid.dz == pytest.approx((6.0 - 2.0) / 1)


def test_missing_required_keys_raise_keyerror():
    base = {
        "nx": 4, "ny": 4, "nz": 4,
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0,
    }

    for key in list(base.keys()):
        bad = dict(base)
        del bad[key]
        with pytest.raises(KeyError):
            initialize_grid(bad)


def test_grid_dimensions_must_be_positive():
    domain = {
        "nx": 0, "ny": 1, "nz": 1,
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0,
    }
    with pytest.raises(ValueError):
        initialize_grid(domain)

    domain["nx"] = -5
    with pytest.raises(ValueError):
        initialize_grid(domain)


def test_extents_must_be_finite():
    bad_values = [float("inf"), float("nan")]

    for bad in bad_values:
        domain = {
            "nx": 1, "ny": 1, "nz": 1,
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
        }
        domain["x_min"] = bad
        with pytest.raises(ValueError):
            initialize_grid(domain)


def test_extents_must_be_ordered_correctly():
    domain = {
        "nx": 1, "ny": 1, "nz": 1,
        "x_min": 0.0, "x_max": 0.0,  # invalid
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0,
    }
    with pytest.raises(ValueError):
        initialize_grid(domain)

    domain["x_max"] = -1.0  # also invalid
    with pytest.raises(ValueError):
        initialize_grid(domain)


def test_dx_dy_dz_must_be_positive_and_finite():
    # dx = (x_max - x_min) / nx
    domain = {
        "nx": 1, "ny": 1, "nz": 1,
        "x_min": 0.0, "x_max": 0.0,  # dx = 0 → invalid
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0,
    }
    with pytest.raises(ValueError):
        initialize_grid(domain)
