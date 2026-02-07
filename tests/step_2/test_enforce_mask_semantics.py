# tests/step_2/test_enforce_mask_semantics.py

import numpy as np
import pytest

from src.step2.enforce_mask_semantics import enforce_mask_semantics


def _make_state(mask: np.ndarray) -> dict:
    """
    Build a minimal Step‑1‑schema‑compliant state for mask semantics tests.
    Only the 'fields' block is required for enforce_mask_semantics.
    """
    return {
        "fields": {
            "Mask": np.asarray(mask)
        }
    }


def test_enforce_mask_semantics_valid_tristate():
    mask = np.array([[[1, 0, -1]]], dtype=int)
    state = _make_state(mask)
    enforce_mask_semantics(state)  # should not raise


def test_enforce_mask_semantics_invalid_positive():
    mask = np.array([[[2]]], dtype=int)
    state = _make_state(mask)
    with pytest.raises(ValueError):
        enforce_mask_semantics(state)


def test_enforce_mask_semantics_invalid_negative():
    mask = np.array([[[-5]]], dtype=int)
    state = _make_state(mask)
    with pytest.raises(ValueError):
        enforce_mask_semantics(state)


def test_enforce_mask_semantics_mixed_invalid():
    mask = np.array([[[0, 1, -1, 2, 3]]], dtype=int)
    state = _make_state(mask)
    with pytest.raises(ValueError):
        enforce_mask_semantics(state)


def test_enforce_mask_semantics_no_fluid_cells():
    mask = np.zeros((2, 2, 2), dtype=int)
    state = _make_state(mask)
    with pytest.raises(ValueError):
        enforce_mask_semantics(state)


def test_enforce_mask_semantics_float_mask_rejected():
    mask = np.array([[[1.0, 0.0, -1.0]]], dtype=float)
    state = _make_state(mask)
    with pytest.raises(ValueError):
        enforce_mask_semantics(state)
