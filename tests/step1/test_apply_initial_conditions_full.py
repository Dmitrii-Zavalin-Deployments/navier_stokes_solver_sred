# tests/step1/test_apply_initial_conditions_full.py

import pytest
import numpy as np

from src.step1.apply_initial_conditions import apply_initial_conditions
from src.step1.types import Fields


def test_apply_initial_conditions_nan_pressure():
    fields = Fields(
        P=np.zeros((2,2,2)),
        U=np.zeros((3,2,2)),
        V=np.zeros((2,3,2)),
        W=np.zeros((2,2,3)),
        Mask=np.zeros((2,2,2)),
    )

    init = {
        "initial_pressure": float("nan"),
        "initial_velocity": [0.0, 0.0, 0.0],
    }

    with pytest.raises(ValueError):
        apply_initial_conditions(fields, init)
