# tests/step1/test_allocate_staggered_fields_math.py

import numpy as np
from src.step1.allocate_staggered_fields import allocate_staggered_fields
from src.step1.types import GridConfig


def make_cfg(nx, ny, nz):
    """
    Minimal valid GridConfig for mathematical tests.
    Domain extents are arbitrary but consistent.
    """
    return GridConfig(
        nx=nx, ny=ny, nz=nz,
        dx=1.0, dy=1.0, dz=1.0,
        x_min=0.0, y_min=0.0, z_min=0.0,
        x_max=float(nx), y_max=float(ny), z_max=float(nz),
    )


def test_staggered_field_shapes_are_correct():
    """
    MAC-grid staggering is mathematically essential.
    This test ensures U, V, W, P, Mask have correct shapes.
    """
    cfg = make_cfg(4, 5, 6)
    fields = allocate_staggered_fields(cfg)

    assert fields.P.shape == (4, 5, 6)
    assert fields.U.shape == (4 + 1, 5, 6)
    assert fields.V.shape == (4, 5 + 1, 6)
    assert fields.W.shape == (4, 5, 6 + 1)
    assert fields.Mask.shape == (4, 5, 6)


def test_fields_are_zero_initialized():
    """
    All velocity and pressure fields must start at zero.
    """
    cfg = make_cfg(3, 3, 3)
    fields = allocate_staggered_fields(cfg)

    assert np.all(fields.P == 0.0)
    assert np.all(fields.U == 0.0)
    assert np.all(fields.V == 0.0)
    assert np.all(fields.W == 0.0)


def test_mask_is_initialized_to_fluid():
    """
    Mask must start as all-fluid (value 1).
    """
    cfg = make_cfg(2, 2, 2)
    fields = allocate_staggered_fields(cfg)

    assert np.all(fields.Mask == 1)
    assert fields.Mask.dtype == int


def test_field_dtypes_are_correct():
    """
    Pressure and velocities must be float arrays.
    Mask must be int array.
    """
    cfg = make_cfg(2, 2, 2)
    fields = allocate_staggered_fields(cfg)

    assert fields.P.dtype == float
    assert fields.U.dtype == float
    assert fields.V.dtype == float
    assert fields.W.dtype == float
    assert fields.Mask.dtype == int
