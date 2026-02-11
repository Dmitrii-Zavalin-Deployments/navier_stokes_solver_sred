# tests/step1/test_apply_initial_conditions_math.py

import numpy as np
import pytest

from src.step1.allocate_staggered_fields import allocate_staggered_fields
from src.step1.apply_initial_conditions import apply_initial_conditions
from src.step1.types import GridConfig


def make_cfg(nx, ny, nz):
    return GridConfig(
        nx=nx, ny=ny, nz=nz,
        dx=1.0, dy=1.0, dz=1.0,
        x_min=0.0, y_min=0.0, z_min=0.0,
        x_max=float(nx), y_max=float(ny), z_max=float(nz),
    )


def test_uniform_initial_conditions_are_applied_correctly():
    cfg = make_cfg(3, 3, 3)
    fields = allocate_staggered_fields(cfg)

    ic = {
        "initial_pressure": 7.5,
        "initial_velocity": [1.0, -2.0, 3.5],
    }

    apply_initial_conditions(fields, ic)

    assert np.all(fields.P == 7.5)
    assert np.all(fields.U == 1.0)
    assert np.all(fields.V == -2.0)
    assert np.all(fields.W == 3.5)


def test_missing_initial_pressure_raises():
    cfg = make_cfg(2, 2, 2)
    fields = allocate_staggered_fields(cfg)

    ic = {"initial_velocity": [0, 0, 0]}

    with pytest.raises(KeyError):
        apply_initial_conditions(fields, ic)


def test_missing_initial_velocity_raises():
    cfg = make_cfg(2, 2, 2)
    fields = allocate_staggered_fields(cfg)

    ic = {"initial_pressure": 1.0}

    with pytest.raises(KeyError):
        apply_initial_conditions(fields, ic)


def test_velocity_vector_must_have_length_3():
    cfg = make_cfg(2, 2, 2)
    fields = allocate_staggered_fields(cfg)

    ic = {
        "initial_pressure": 1.0,
        "initial_velocity": [1.0, 2.0],  # wrong length
    }

    with pytest.raises(ValueError):
        apply_initial_conditions(fields, ic)


def test_pressure_must_be_finite():
    cfg = make_cfg(2, 2, 2)
    fields = allocate_staggered_fields(cfg)

    ic = {
        "initial_pressure": float("inf"),
        "initial_velocity": [0.0, 0.0, 0.0],
    }

    with pytest.raises(ValueError):
        apply_initial_conditions(fields, ic)


def test_velocity_components_must_be_finite():
    cfg = make_cfg(2, 2, 2)
    fields = allocate_staggered_fields(cfg)

    ic = {
        "initial_pressure": 1.0,
        "initial_velocity": [0.0, float("nan"), 0.0],
    }

    with pytest.raises(ValueError):
        apply_initial_conditions(fields, ic)
