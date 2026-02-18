# tests/step1/test_allocate_fields.py

import numpy as np
from src.step1.allocate_fields import allocate_fields
from src.step1.types import GridConfig


def test_cell_centered_field_shapes():
    grid = GridConfig(
        nx=4, ny=5, nz=6,
        dx=1.0, dy=1.0, dz=1.0,
        x_min=0.0, x_max=1.0,
        y_min=0.0, y_max=1.0,
        z_min=0.0, z_max=1.0,
    )

    fields = allocate_fields(grid)

    expected_shape = (4, 5, 6)
    for name in ("U", "V", "W", "P"):
        arr = fields[name]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == expected_shape
