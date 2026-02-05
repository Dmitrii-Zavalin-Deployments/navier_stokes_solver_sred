# tests/step2/test_precompute_constants_and_masks.py
import numpy as np
import pytest

from step2.precompute_constants import precompute_constants
from step2.create_fluid_mask import create_fluid_mask


class DummyState:
    def __init__(self, mask, constants=None, dx=0.1, dy=0.2, dz=0.3, dt=0.01, rho=1.0, mu=0.1):
        self.Mask = np.array(mask)
        self.constants = constants
        self.grid = type(
            "Grid",
            (),
            dict(dx=dx, dy=dy, dz=dz),
        )()
        self.config = type(
            "Config",
            (),
            dict(
                fluid_properties={"density": rho, "viscosity": mu},
                simulation_parameters={"dt": dt},
            ),
        )()


def test_precompute_constants_normal():
    state = DummyState(mask=np.zeros((2, 2, 2), dtype=int), dx=0.1, dy=0.2, dz=0.3, dt=0.01)
    constants = precompute_constants(state)
    assert constants["dx"] == pytest.approx(0.1)
    assert constants["inv_dx"] == pytest.approx(10.0)
    assert constants["inv_dx2"] == pytest.approx(100.0)


def test_precompute_constants_very_small_dx():
    state = DummyState(mask=np.zeros((1, 1, 1), dtype=int), dx=1e-12, dy=1e-12, dz=1e-12, dt=0.01)
    constants = precompute_constants(state)
    assert np.isfinite(constants["inv_dx"])
    assert np.isfinite(constants["inv_dx2"])


def test_precompute_constants_dt_zero_rejected():
    state = DummyState(mask=np.zeros((1, 1, 1), dtype=int), dt=0.0)
    with pytest.raises(ValueError) as excinfo:
        precompute_constants(state)
    assert "dt must be > 0" in str(excinfo.value)


def test_precompute_constants_existing_constants_passthrough():
    existing = {"dx": 0.1, "inv_dx": 10.0}
    state = DummyState(mask=np.zeros((1, 1, 1), dtype=int), constants=existing)
    constants = precompute_constants(state)
    assert constants is existing


def test_create_fluid_mask_mixed():
    mask = np.array(
        [
            [[0, 1]],
            [[-1, 0]],
        ],
        dtype=int,
    )
    state = DummyState(mask=mask)
    is_fluid, is_boundary = create_fluid_mask(state)
    assert is_fluid.shape == mask.shape
    assert is_boundary.shape == mask.shape
    assert is_fluid[0, 0, 1] and is_fluid[1, 0, 0]
    assert is_boundary[1, 0, 0]
    assert not is_fluid[0, 0, 0]


def test_create_fluid_mask_all_fluid():
    mask = np.ones((2, 2, 2), dtype=int)
    state = DummyState(mask=mask)
    is_fluid, is_boundary = create_fluid_mask(state)
    assert np.all(is_fluid)
    assert not np.any(is_boundary)


def test_create_fluid_mask_all_boundary_fluid():
    mask = -np.ones((2, 2, 2), dtype=int)
    state = DummyState(mask=mask)
    is_fluid, is_boundary = create_fluid_mask(state)
    assert np.all(is_fluid)
    assert np.all(is_boundary)


def test_create_fluid_mask_shape_preserved():
    mask = np.zeros((3, 4, 5), dtype=int)
    state = DummyState(mask=mask)
    is_fluid, is_boundary = create_fluid_mask(state)
    assert is_fluid.shape == (3, 4, 5)
    assert is_boundary.shape == (3, 4, 5)


def test_create_fluid_mask_float_rejected():
    mask = np.ones((2, 2, 2), dtype=float)
    state = DummyState(mask=mask)
    with pytest.raises(ValueError) as excinfo:
        create_fluid_mask(state)
    assert "integer dtype" in str(excinfo.value)
