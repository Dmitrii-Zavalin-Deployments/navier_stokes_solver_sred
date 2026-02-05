# tests/step2/test_enforce_mask_semantics.py
import numpy as np
import pytest

from step2.enforce_mask_semantics import enforce_mask_semantics


class DummyState:
    def __init__(self, mask):
        self.Mask = np.array(mask)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((2, 2, 2), dtype=int),
        np.ones((3, 3, 3), dtype=int),
        np.array([[[0, 1], [-1, 0]]], dtype=int),
    ],
)
def test_enforce_mask_semantics_valid_tristate(mask):
    state = DummyState(mask)
    # should not raise
    enforce_mask_semantics(state)


@pytest.mark.parametrize(
    "mask, invalid_values",
    [
        (np.array([[[0, 2]]], dtype=int), [2]),
        (np.array([[[-5, 0]]], dtype=int), [-5]),
        (np.array([[[0, 1, -1, 2, 3]]], dtype=int), [2, 3]),
    ],
)
def test_enforce_mask_semantics_invalid_values(mask, invalid_values):
    state = DummyState(mask)
    with pytest.raises(ValueError) as excinfo:
        enforce_mask_semantics(state)
    for v in invalid_values:
        assert str(v) in str(excinfo.value)


def test_enforce_mask_semantics_no_fluid_cells():
    mask = np.zeros((2, 2, 2), dtype=int)
    state = DummyState(mask)
    with pytest.raises(ValueError) as excinfo:
        enforce_mask_semantics(state)
    assert "no fluid or boundary-fluid" in str(excinfo.value)


@pytest.mark.parametrize("dtype", [float, np.float64])
def test_enforce_mask_semantics_float_mask_rejected(dtype):
    mask = np.array([[[1.0, 0.0, -1.0]]], dtype=dtype)
    state = DummyState(mask)
    with pytest.raises(ValueError) as excinfo:
        enforce_mask_semantics(state)
    assert "integer dtype" in str(excinfo.value)
