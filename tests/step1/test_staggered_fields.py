# tests/step1/test_staggered_fields.py
import pytest
from src.step1.allocate_staggered_fields import allocate_staggered_fields
from src.step1.types import GridConfig


def test_staggered_field_shapes():
    grid = GridConfig(
        nx=10, ny=20, nz=30,
        dx=1.0, dy=1.0, dz=1.0,
        x_min=0, x_max=1,
        y_min=0, y_max=1,
        z_min=0, z_max=1,
    )

    fields = allocate_staggered_fields(grid)

    assert fields.U.shape == (11, 20, 30)
    assert fields.V.shape == (10, 21, 30)
    assert fields.W.shape == (10, 20, 31)


def test_zero_or_negative_grid():
    with pytest.raises(ValueError):
        GridConfig(
            nx=0, ny=10, nz=10,
            dx=1, dy=1, dz=1,
            x_min=0, x_max=1,
            y_min=0, y_max=1,
            z_min=0, z_max=1,
        )
