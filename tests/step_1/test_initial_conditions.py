# tests/step1/test_initial_conditions.py
import numpy as np
import pytest

from src.step1.allocate_staggered_fields import allocate_staggered_fields
from src.step1.apply_initial_conditions import apply_initial_conditions
from src.step1.types import GridConfig


def make_grid():
    return GridConfig(
        nx=4, ny=4, nz=4,
        dx=1, dy=1, dz=1,
        x_min=0, x_max=1,
        y_min=0, y_max=1,
        z_min=0, z_max=1,
    )


def test_uniform_velocity_broadcast():
    grid = make_grid()
    fields = allocate_staggered_fields(grid)

    init = {"initial_velocity": [1.0, 0.0, -0.5], "initial_pressure": 2.0}
    apply_initial_conditions(fields, init)

    assert np.all(fields.U == 1.0)
    assert np.all(fields.V == 0.0)
    assert np.all(fields.W == -0.5)
    assert np.all(fields.P == 2.0)


def test_non_finite_velocity():
    grid = make_grid()
    fields = allocate_staggered_fields(grid)

    bad_init = {"initial_velocity": [float("nan"), 1.0, 0.0], "initial_pressure": 1.0}

    with pytest.raises(ValueError):
        apply_initial_conditions(fields, bad_init)
